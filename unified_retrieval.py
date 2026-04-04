from typing import Dict, Optional

import torch
import torch.nn as nn

from generative_retrieval import OneRecVDecoderBlock
from two_tower_basic import TwoTowerBasic


class UnifiedRetrieval(TwoTowerBasic):
    """Unified model: train tower loss + semantic generation loss together.

    This demonstrates the core repository claim: generative retrieval does not need to be
    developed in isolation. We can jointly optimize:
    1) sampled-softmax retrieval loss from the two-tower branch
    2) teacher-forcing semantic generation loss from the decoder branch

    At inference time, we can call whichever path we want:
    - tower path: fast matmul retrieval over cached item embeddings
    - generative path: decode semantic IDs token by token

    Unified ``train_forward`` API:
    - used: ``user_sequence_features``, ``user_static_features``, ``item_static_features``,
      ``semantic_ids``, optional ``reward_weights``
    - ignored: ``cluster_ids``, ``candidate_item_static_features``
    """

    def __init__(
        self,
        user_vocab_size: int,
        user_sequence_length: int,
        user_embedding_dim: int,
        user_static_dim: int,
        item_static_dim: int,
        hidden_dim: int = 128,
        oob_pool_size: int = 4096,
        oob_negative_count: int = 4,
        dropout: float = 0.1,
        semantic_vocab_size: int = 4096,
        semantic_seq_len: int = 4,
        user_static_token_count: int = 2,
        decoder_layers: int = 3,
        decoder_heads: int = 4,
        decoder_ffn: int = 256,
        moe_experts: int = 4,
        tower_loss_weight: float = 1.0,
        generation_loss_weight: float = 1.0,
    ) -> None:
        super().__init__(
            user_vocab_size=user_vocab_size,
            user_sequence_length=user_sequence_length,
            user_embedding_dim=user_embedding_dim,
            user_static_dim=user_static_dim,
            item_static_dim=item_static_dim,
            hidden_dim=hidden_dim,
            oob_pool_size=oob_pool_size,
            oob_negative_count=oob_negative_count,
            dropout=dropout,
        )
        self.semantic_seq_len = semantic_seq_len
        self.user_static_token_count = user_static_token_count
        self.tower_loss_weight = tower_loss_weight
        self.generation_loss_weight = generation_loss_weight

        self.user_token_projection = nn.Linear(user_embedding_dim, hidden_dim)
        self.user_static_token_projection = nn.Linear(
            user_static_dim, user_static_token_count * hidden_dim
        )
        self.user_positional = nn.Parameter(
            torch.randn(user_sequence_length + user_static_token_count, hidden_dim) * 0.02
        )
        self.semantic_embedding = nn.Embedding(semantic_vocab_size, hidden_dim)
        self.semantic_positional = nn.Parameter(
            torch.randn(semantic_seq_len, hidden_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                OneRecVDecoderBlock(
                    hidden_dim=hidden_dim,
                    num_heads=decoder_heads,
                    ffn_dim=decoder_ffn,
                    num_experts=moe_experts,
                    dropout=dropout,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.output_proj = nn.Linear(hidden_dim, semantic_vocab_size)
        self.semantic_loss_fn = nn.CrossEntropyLoss(reduction="none")

    def _build_static_tokens(self, user_static_features: torch.Tensor) -> torch.Tensor:
        """Convert static user inputs into static tokens.

        Args:
            user_static_features: Static features ``[B, F]`` or static tokens
                ``[B, S_static, hidden_dim]``.

        Returns:
            Static tokens with shape ``[B, S_static, hidden_dim]``.
        """
        if user_static_features.dim() == 3:
            if user_static_features.size(-1) != self.user_token_projection.out_features:
                raise ValueError(
                    "3D user_static_features must have last dimension equal to hidden_dim."
                )
            return user_static_features
        if user_static_features.dim() != 2:
            raise ValueError("user_static_features must be [B, F] or [B, S, D].")
        projected = self.user_static_token_projection(user_static_features)
        return projected.view(
            user_static_features.size(0),
            self.user_static_token_count,
            self.user_token_projection.out_features,
        )

    def _encode_user_tokens(
        self,
        user_sequence_features: torch.Tensor,
        user_static_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode users into decoder cross-attention memory tokens.

        Args:
            user_sequence_features: Sequence token ids with shape ``[B, L]``.
            user_static_features: Static features ``[B, F]`` or static tokens
                ``[B, S_static, hidden_dim]``.

        Returns:
            User memory tokens with shape ``[B, L + S_static, hidden_dim]``.
        """
        # sequence: [B, L] -> [B, L, user_embedding_dim] -> [B, L, hidden_dim]
        seq_embeds = self.user_embedding(user_sequence_features)
        seq_hidden = self.user_token_projection(seq_embeds)
        # static: [B, F] or [B, S_static, D] -> [B, S_static, hidden_dim]
        static_tokens = self._build_static_tokens(user_static_features)
        # concat tokens on dim=1 => [B, L + S_static, hidden_dim]
        user_memory = torch.cat([seq_hidden, static_tokens], dim=1)
        if user_memory.size(1) > self.user_positional.size(0):
            raise ValueError(
                f"User token length {user_memory.size(1)} exceeds positional table "
                f"size {self.user_positional.size(0)}"
            )
        user_memory = user_memory + self.user_positional[: user_memory.size(1)].unsqueeze(0)
        return user_memory

    def _causal_mask(self, target_len: int, device: torch.device) -> torch.Tensor:
        """Build a causal self-attention mask.

        Args:
            target_len: Number of generated steps ``T``.
            device: Device where the mask should be allocated.

        Returns:
            Boolean causal mask with shape ``[T, T]``.
        """
        return torch.triu(
            torch.ones(target_len, target_len, dtype=torch.bool, device=device), diagonal=1
        )

    def _decode_teacher_forcing(
        self,
        user_memory: torch.Tensor,
        semantic_ids: torch.Tensor,
        reward_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute semantic cross-entropy loss using teacher forcing.

        Args:
            user_memory: Decoder memory tokens with shape ``[B, L_user, hidden_dim]``.
            semantic_ids: Semantic token ids with shape ``[B, S]`` including BOS.
            reward_weights: Optional per-example weights with shape ``[B]``.

        Returns:
            Scalar semantic generation loss tensor.
        """
        device = semantic_ids.device
        if semantic_ids.size(1) != self.semantic_seq_len:
            raise ValueError(
                f"Expected semantic_ids with length {self.semantic_seq_len}, got {semantic_ids.size(1)}"
            )
        decoder_input = semantic_ids[:, :-1]
        decoder_target = semantic_ids[:, 1:]
        # decoder_input/decoder_target: [B, S-1]
        generated = self.semantic_embedding(decoder_input)
        generated = generated + self.semantic_positional[: decoder_input.size(1)].unsqueeze(0)
        # generated: [B, S-1, hidden_dim]
        causal_mask = self._causal_mask(generated.size(1), device)
        x = generated
        for block in self.blocks:
            x = block(x, user_memory, causal_mask)
        logits = self.output_proj(x)
        # logits: [B, S-1, semantic_vocab_size]
        token_losses = self.semantic_loss_fn(
            logits.reshape(-1, logits.size(-1)),
            decoder_target.reshape(-1),
        ).view(logits.size(0), logits.size(1))
        # token_losses: [B, S-1]
        if reward_weights is not None:
            token_weights = reward_weights.to(device).unsqueeze(1).expand_as(token_losses)
            return (token_losses * token_weights).sum() / token_weights.sum().clamp_min(1e-6)
        return token_losses.mean()

    def train_forward(
        self,
        user_sequence_features: torch.Tensor,
        user_static_features: torch.Tensor,
        item_static_features: torch.Tensor,
        reward_weights: Optional[torch.Tensor] = None,
        cluster_ids: Optional[torch.Tensor] = None,
        semantic_ids: Optional[torch.Tensor] = None,
        candidate_item_static_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the joint retrieval + semantic generation objective.

        Args:
            user_sequence_features: Sequence token ids with shape ``[B, L]``.
            user_static_features: Static features ``[B, F]`` or static tokens
                ``[B, S_static, hidden_dim]``.
            item_static_features: Positive item features with shape ``[B, F2]``.
            reward_weights: Optional per-example weights with shape ``[B]``.
            cluster_ids: Unused in this class.
            semantic_ids: Semantic token targets with shape ``[B, S]`` (required).
            candidate_item_static_features: Unused in this class.

        Returns:
            Dictionary with scalar tensors: ``loss``, ``tower_loss``,
            and ``generation_loss``.
        """
        del cluster_ids, candidate_item_static_features
        if semantic_ids is None:
            raise ValueError("semantic_ids is required for UnifiedRetrieval.")
        # Tower branch (inherited): [B, L]/[B, F1]/[B, F2] -> sampled-softmax loss.
        tower_loss = super().train_forward(
            user_sequence_features=user_sequence_features,
            user_static_features=user_static_features,
            item_static_features=item_static_features,
            reward_weights=reward_weights,
        )
        # Generative branch: user memory tokens + teacher forcing semantic CE.
        user_memory = self._encode_user_tokens(user_sequence_features, user_static_features)
        generation_loss = self._decode_teacher_forcing(
            user_memory=user_memory,
            semantic_ids=semantic_ids,
            reward_weights=reward_weights,
        )
        total_loss = (
            self.tower_loss_weight * tower_loss
            + self.generation_loss_weight * generation_loss
        )
        return {
            "loss": total_loss,
            "tower_loss": tower_loss,
            "generation_loss": generation_loss,
        }

    @torch.no_grad()
    def generate_semantic_ids(
        self,
        user_sequence_features: torch.Tensor,
        user_static_features: torch.Tensor,
        bos_token_id: int,
    ) -> torch.Tensor:
        """Generate semantic ids greedily from the unified decoder branch.

        Args:
            user_sequence_features: Sequence token ids with shape ``[B, L]``.
            user_static_features: Static features ``[B, F]`` or static tokens
                ``[B, S_static, hidden_dim]``.
            bos_token_id: Token id to seed generation at position 0.

        Returns:
            Generated semantic id sequences with shape ``[B, semantic_seq_len]``.
        """
        device = user_sequence_features.device
        user_memory = self._encode_user_tokens(user_sequence_features, user_static_features)
        batch_size = user_sequence_features.size(0)
        generated_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=device
        )
        for step in range(self.semantic_seq_len - 1):
            x = self.semantic_embedding(generated_ids)
            x = x + self.semantic_positional[: x.size(1)].unsqueeze(0)
            causal_mask = self._causal_mask(x.size(1), device)
            for block in self.blocks:
                x = block(x, user_memory, causal_mask)
            next_logits = self.output_proj(x[:, -1, :])
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            if step + 1 >= self.semantic_seq_len - 1:
                break
        return generated_ids

    def retrieve_with_tower(
        self,
        user_sequence_features: torch.Tensor,
        user_static_features: torch.Tensor,
        topk: int = 10,
        return_all_scores: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run the two-tower inference path over cached item embeddings.

        Args:
            user_sequence_features: Sequence token ids with shape ``[B, L]``.
            user_static_features: Static user features with shape ``[B, F1]``.
            topk: Number of items to return.
            return_all_scores: If ``True``, include full score matrix.

        Returns:
            Retrieval dictionary from ``TwoTowerBasic.retrieve``.
        """
        return self.retrieve(
            user_sequence_features=user_sequence_features,
            user_static_features=user_static_features,
            topk=topk,
            return_all_scores=return_all_scores,
        )
