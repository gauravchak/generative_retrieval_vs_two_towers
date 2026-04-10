from typing import Optional

import torch
import torch.nn as nn

from two_tower_basic import TwoTowerBasic


class MultiHeadTwoTower(TwoTowerBasic):
    """ColBERT-style late-interaction retriever that specializes the basic tower.

    This subclass only updates the user encoding to produce ``(B, H, head_dim)`` heads,
    replaces the item encoder with a lighter projection to ``head_dim``, and aggregates the
    per-head logits before reusing the base OOB sampler + loss infrastructure.

    Unified ``train_forward`` API:
    - used: ``user_sequence_features``, ``user_static_features``, ``item_static_features``,
      optional ``reward_weights``
    - ignored: ``cluster_ids``, ``semantic_ids``, ``candidate_item_static_features``
    """

    def __init__(
        self,
        user_vocab_size: int,
        user_sequence_length: int,
        user_embedding_dim: int,
        user_static_dim: int,
        item_static_dim: int,
        hidden_dim: int = 128,
        head_dim: int = 64,
        num_heads: int = 4,
        aggregator: str = "max",
        oob_negative_count: int = 4,
        oob_pool_size: int = 4096,
        dropout: float = 0.1,
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
        assert aggregator in {
            "max",
            "softmax",
        }, "aggregator must be max or softmax"
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.aggregator = aggregator
        # Heads project ``(B, hidden_dim*2)`` into ``(B, H * head_dim)`` before reshaping.
        self.user_heads = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads * head_dim),
        )
        # Override the item encoder to emit ``(B, head_dim)`` vectors for the late interaction.
        self.item_encoder = nn.Sequential(
            nn.Linear(item_static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, head_dim),
        )

    def _encode_user(
        self,
        user_sequence_features: torch.Tensor,
        user_static_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode users into multiple interaction heads.

        Args:
            user_sequence_features: Sequence token ids with shape ``[B, L]``.
            user_static_features: Static user features with shape ``[B, F1]``.

        Returns:
            Multi-head user embeddings with shape ``[B, H, head_dim]``.
        """
        # Sequence pipeline reuses the base embeddings from TwoTowerBasic.
        embeds = self.user_embedding(user_sequence_features)
        # embeds: [B, L, D]
        seq_flat = embeds.reshape(embeds.size(0), -1)
        # seq_flat: [B, L*D]
        seq_repr = self.user_sequence_projection(seq_flat)
        # seq_repr: [B, hidden_dim]
        static_repr = self.user_static_projection(user_static_features)
        # static_repr: [B, hidden_dim]
        fused = torch.cat([seq_repr, static_repr], dim=-1)
        heads = self.user_heads(fused)
        # heads reshaped to [B, H, head_dim]
        return heads.view(heads.size(0), self.num_heads, self.head_dim)

    def _aggregate_logits(self, head_scores: torch.Tensor) -> torch.Tensor:
        """Aggregate per-head candidate logits into a single candidate score.

        Args:
            head_scores: Head-wise scores with shape ``[B, H, 1 + num_oob]``.

        Returns:
            Aggregated candidate logits with shape ``[B, 1 + num_oob]``.
        """
        # head_scores: [B, H, 1 + num_oob], aggregator returns [B, 1 + num_oob]
        if self.aggregator == "max":
            return head_scores.max(dim=1).values
        return torch.logsumexp(head_scores, dim=1)

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
        """Compute multi-head sampled-softmax retrieval loss.

        Args:
            user_sequence_features: Sequence token ids with shape ``[B, L]``.
            user_static_features: Static user features with shape ``[B, F1]``.
            item_static_features: Positive item features with shape ``[B, F2]``.
            reward_weights: Optional per-example weights with shape ``[B]``.
            cluster_ids: Unused in this class.
            semantic_ids: Unused in this class.
            candidate_item_static_features: Unused in this class.

        Returns:
            Scalar training loss tensor.
        """
        del cluster_ids, semantic_ids, candidate_item_static_features
        device = user_static_features.device
        user_heads = self._encode_user(
            user_sequence_features, user_static_features
        )
        # user_heads: [B, H, head_dim]

        candidates = self.oob_sampler.get_candidates(item_static_features)
        candidate_repr = self._encode_items(candidates)
        # candidate_repr: [B, 1 + num_oob, head_dim]

        head_scores = torch.einsum("bhd,bnd->bhn", user_heads, candidate_repr)
        # head_scores: [B, H, 1 + num_oob]
        logits = self._aggregate_logits(head_scores)
        # logits: [B, 1 + num_oob]

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
        # labels: [B], positive index is 0
        losses = self.loss_fn(logits, labels)
        if reward_weights is not None:
            weights = reward_weights.to(device)
            valid_mask = weights > 0
            valid_weights = weights[valid_mask]
            total_weight = valid_weights.sum()
            if total_weight < 1:
                loss = losses.sum() * 0.0
            else:
                loss = (losses[valid_mask] * valid_weights).sum() / total_weight
        else:
            loss = losses.mean()
        return loss
