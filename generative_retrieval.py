from typing import Optional

import torch
import torch.nn as nn

from two_tower_basic import TwoTowerBasic


class MoEFFN(nn.Module):
    """Small dense MoE-FFN used inside each decoder block."""

    def __init__(
        self, hidden_dim: int, ffn_dim: int, num_experts: int, dropout: float
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated mixture-of-experts feed-forward transformation.

        Args:
            x: Decoder hidden states with shape ``[B, T, D]``.

        Returns:
            MoE output with shape ``[B, T, D]``.
        """
        # x: [B, T, D]
        gate_probs = torch.softmax(self.gate(x), dim=-1)
        # gate_probs: [B, T, num_experts]
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=-1
        )
        # expert_outputs: [B, T, D, num_experts]
        mixed = torch.sum(expert_outputs * gate_probs.unsqueeze(2), dim=-1)
        # mixed: [B, T, D]
        return self.dropout(mixed)


class OneRecVDecoderBlock(nn.Module):
    """OneRecV-style ordering: cross-attn (user) -> self-attn (generated) -> MoE-FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_experts: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.moe_ffn = MoEFFN(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            dropout=dropout,
        )
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        generated_tokens: torch.Tensor,
        user_memory: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run one decoder block in cross-attn -> self-attn -> MoE order.

        Args:
            generated_tokens: Generated-token states ``[B, T, D]``.
            user_memory: User memory tokens for cross-attn ``[B, L_user, D]``.
            causal_mask: Causal attention mask for self-attn ``[T, T]``.

        Returns:
            Updated generated-token states with shape ``[B, T, D]``.
        """
        # generated_tokens: [B, T, D], user_memory: [B, L, D]
        cross_out, _ = self.cross_attn(
            query=generated_tokens,
            key=user_memory,
            value=user_memory,
            need_weights=False,
        )
        x = self.norm_cross(generated_tokens + self.dropout(cross_out))

        self_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = self.norm_self(x + self.dropout(self_out))

        ffn_out = self.moe_ffn(x)
        x = self.norm_ffn(x + ffn_out)
        return x


class GenerativeRetrieval(TwoTowerBasic):
    """Generative retrieval with lightweight user encoder and OneRecV-style decoder blocks.

    Signature mirrors ``TwoTowerBasic.train_forward`` but adds ``semantic_ids: [B, S]``.
    ``semantic_ids`` is expected to include ``[BOS]`` so teacher forcing can use
    inputs ``[:, :-1]`` and targets ``[:, 1:]``.

    Unified ``train_forward`` API:
    - used: ``user_sequence_features``, ``user_static_features``, ``semantic_ids``,
      optional ``reward_weights``
    - ignored: ``item_static_features``, ``cluster_ids``, ``candidate_item_static_features``
    """

    def __init__(
        self,
        user_vocab_size: int,
        user_sequence_length: int,
        user_embedding_dim: int,
        user_static_dim: int,
        item_static_dim: int,
        hidden_dim: int = 128,
        semantic_vocab_size: int = 4096,
        semantic_seq_len: int = 4,
        user_static_token_count: int = 2,
        decoder_layers: int = 3,
        decoder_heads: int = 4,
        decoder_ffn: int = 256,
        moe_experts: int = 4,
        dropout: float = 0.1,
    ) -> None:
        # We still inherit from the basic tower for consistency, but we do not use
        # the sampled-softmax branch for the generative objective.
        super().__init__(
            user_vocab_size=user_vocab_size,
            user_sequence_length=user_sequence_length,
            user_embedding_dim=user_embedding_dim,
            user_static_dim=user_static_dim,
            item_static_dim=item_static_dim,
            hidden_dim=hidden_dim,
            oob_pool_size=1,
            oob_negative_count=1,
            dropout=dropout,
        )
        self.semantic_seq_len = semantic_seq_len
        self.user_static_token_count = user_static_token_count
        self.user_token_projection = nn.Linear(user_embedding_dim, hidden_dim)
        self.user_static_token_projection = nn.Linear(
            user_static_dim, user_static_token_count * hidden_dim
        )
        self.user_positional = nn.Parameter(
            torch.randn(
                user_sequence_length + user_static_token_count, hidden_dim
            )
            * 0.02
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

    def _encode_user_tokens(
        self,
        user_sequence_features: torch.Tensor,
        user_static_features: torch.Tensor,
    ) -> torch.Tensor:
        """Create user memory tokens used by decoder cross-attention.

        Args:
            user_sequence_features: Sequence token ids with shape ``[B, L]``.
            user_static_features: Static features ``[B, F]`` or static tokens
                ``[B, S_static, D]``.

        Returns:
            User memory tokens with shape ``[B, L + S_static, hidden_dim]``.
        """
        seq_embeds = self.user_embedding(user_sequence_features)
        # seq_embeds: [B, L, user_embedding_dim]
        seq_hidden = self.user_token_projection(seq_embeds)
        # seq_hidden: [B, L, hidden_dim]
        static_tokens = self._build_static_tokens(user_static_features)
        # static_tokens: [B, S_static, hidden_dim]
        user_memory = torch.cat([seq_hidden, static_tokens], dim=1)
        # user_memory: [B, L + S_static, hidden_dim]
        if user_memory.size(1) > self.user_positional.size(0):
            raise ValueError(
                f"User token length {user_memory.size(1)} exceeds positional table "
                f"size {self.user_positional.size(0)}"
            )
        user_memory = user_memory + self.user_positional[
            : user_memory.size(1)
        ].unsqueeze(0)
        # user_memory: [B, L + S_static, hidden_dim]
        return user_memory

    def _build_static_tokens(
        self, user_static_features: torch.Tensor
    ) -> torch.Tensor:
        """Convert static features into static token embeddings.

        Args:
            user_static_features: Static features ``[B, F_static]`` or pre-tokenized
                static tokens ``[B, S_static, hidden_dim]``.

        Returns:
            Static tokens with shape ``[B, S_static, hidden_dim]``.
        """
        if user_static_features.dim() == 3:
            if (
                user_static_features.size(-1)
                != self.user_token_projection.out_features
            ):
                raise ValueError(
                    "3D user_static_features must have last dimension equal to hidden_dim."
                )
            return user_static_features
        if user_static_features.dim() != 2:
            raise ValueError(
                "user_static_features must be [B, F] or [B, S, D]."
            )
        projected = self.user_static_token_projection(user_static_features)
        return projected.view(
            user_static_features.size(0),
            self.user_static_token_count,
            self.user_token_projection.out_features,
        )

    def _causal_mask(
        self, target_len: int, device: torch.device
    ) -> torch.Tensor:
        """Build a causal mask for decoder self-attention.

        Args:
            target_len: Number of generated time steps ``T``.
            device: Device where the mask should be allocated.

        Returns:
            Boolean mask with shape ``[T, T]`` where ``True`` entries are masked.
        """
        return torch.triu(
            torch.ones(target_len, target_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def train_forward(
        self,
        user_sequence_features: torch.Tensor,
        user_static_features: torch.Tensor,
        item_static_features: torch.Tensor,
        reward_weights: Optional[torch.Tensor] = None,
        cluster_ids: Optional[torch.Tensor] = None,
        semantic_ids: Optional[torch.Tensor] = None,
        candidate_item_static_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute teacher-forcing semantic generation loss.

        Args:
            user_sequence_features: Sequence token ids with shape ``[B, L]``.
            user_static_features: Static features ``[B, F]`` or static tokens
                ``[B, S_static, D]``.
            item_static_features: Unused in this class; kept for unified API.
            reward_weights: Optional per-example weights with shape ``[B]``.
            cluster_ids: Unused in this class.
            semantic_ids: Semantic token targets with shape ``[B, S]`` (required).
            candidate_item_static_features: Unused in this class.

        Returns:
            Scalar semantic generation loss tensor.
        """
        del item_static_features, cluster_ids, candidate_item_static_features
        device = user_static_features.device
        if semantic_ids is None:
            raise ValueError(
                "semantic_ids is required for GenerativeRetrieval."
            )
        if semantic_ids.size(1) != self.semantic_seq_len:
            raise ValueError(
                f"Expected semantic_ids with length {self.semantic_seq_len}, "
                f"got {semantic_ids.size(1)}"
            )
        user_memory = self._encode_user_tokens(
            user_sequence_features, user_static_features
        )
        # user_memory: [B, L + S_static, hidden_dim]

        decoder_input = semantic_ids[:, :-1]
        decoder_target = semantic_ids[:, 1:]
        # decoder_input/decoder_target: [B, S-1]

        generated = self.semantic_embedding(decoder_input)
        generated = generated + self.semantic_positional[
            : decoder_input.size(1)
        ].unsqueeze(0)
        # generated: [B, S-1, hidden_dim]

        causal_mask = self._causal_mask(generated.size(1), device)
        x = generated
        for block in self.blocks:
            x = block(x, user_memory, causal_mask)
        # x: [B, S-1, hidden_dim]

        logits = self.output_proj(x)
        # logits: [B, S-1, semantic_vocab_size]
        token_losses = self.semantic_loss_fn(
            logits.reshape(-1, logits.size(-1)),
            decoder_target.reshape(-1),
        ).view(logits.size(0), logits.size(1))
        # token_losses: [B, S-1]

        if reward_weights is not None:
            # reward_weights: [B] -> token-aligned weights [B, S-1]
            token_weights = (
                reward_weights.to(device).unsqueeze(1).expand_as(token_losses)
            )
            valid_mask = token_weights > 0
            valid_weights = token_weights[valid_mask]
            total_weight = valid_weights.sum()
            if total_weight < 1:
                return token_losses.sum() * 0.0
            return (
                token_losses[valid_mask] * valid_weights
            ).sum() / total_weight
        return token_losses.mean()
