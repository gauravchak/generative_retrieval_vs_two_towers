from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from multi_head_two_tower import MultiHeadTwoTower
from two_tower_basic import TwoTowerBasic


class MultiStageRetrieval(TwoTowerBasic):
    """Two-stage retriever (arXiv:2306.04039) that builds directly on ``TwoTowerBasic``.

    By subclassing ``TwoTowerBasic`` we reuse the prefilter-stage encode / OOB sampler /
    cached item-index behavior and only add the overarch logic on top. That keeps the diff
    focused on the late-interaction reranker while still explaining the extra tensors.

    Unified ``train_forward`` API:
    - used: ``user_sequence_features``, ``user_static_features``,
      ``candidate_item_static_features`` (or ``item_static_features`` if already [B, K, F2]),
      optional ``reward_weights``
    - ignored: ``cluster_ids``, ``semantic_ids``
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
        overarch_hidden: int = 128,
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
        self.overarch = MultiHeadTwoTower(
            user_vocab_size=user_vocab_size,
            user_sequence_length=user_sequence_length,
            user_embedding_dim=user_embedding_dim,
            user_static_dim=user_static_dim,
            item_static_dim=item_static_dim,
            hidden_dim=hidden_dim,
            head_dim=head_dim,
            num_heads=num_heads,
            aggregator=aggregator,
            oob_negative_count=oob_negative_count,
            oob_pool_size=oob_pool_size,
            dropout=dropout,
        )
        self.overarch_mlp = nn.Sequential(
            nn.Linear(1 + 2 * head_dim, overarch_hidden),
            nn.ReLU(),
            nn.Linear(overarch_hidden, 1),
        )
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.item_pool_static: Optional[torch.Tensor] = None
        self.item_pool_ids: Optional[torch.Tensor] = None

    def build_item_pool(
        self, item_static_features: torch.Tensor, item_ids: torch.Tensor
    ) -> None:
        """Build stage-1 retrieval cache and retain item statics for stage 2.

        Args:
            item_static_features: Item pool features with shape ``[P, F2]``.
            item_ids: Integer ids aligned to the pool, shape ``[P]``.

        Returns:
            None. Stores both cached embeddings and raw static features.
        """
        super().build_item_index(item_static_features, item_ids)
        self.item_pool_static = item_static_features.detach()
        self.item_pool_ids = item_ids.clone()

    def extend_item_pool(
        self, item_static_features: torch.Tensor, item_ids: torch.Tensor
    ) -> None:
        """Append additional items to stage-1 cache and stage-2 static store.

        Args:
            item_static_features: New item features with shape ``[P_new, F2]``.
            item_ids: Ids aligned to new rows, shape ``[P_new]``.

        Returns:
            None. Updates in-memory item stores in place.
        """
        super().extend_item_index(item_static_features, item_ids)
        if self.item_pool_static is None:
            self.item_pool_static = item_static_features.detach()
        else:
            self.item_pool_static = torch.cat(
                [self.item_pool_static, item_static_features.detach()], dim=0
            )
        if self.item_pool_ids is None:
            self.item_pool_ids = item_ids.clone()
        else:
            self.item_pool_ids = torch.cat(
                [self.item_pool_ids, item_ids.clone()], dim=0
            )

    def _prefilter_scores(
        self,
        user_repr: torch.Tensor,
        candidate_item_static_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute prefilter-stage scores for candidate items.

        Args:
            user_repr: Prefilter user embeddings with shape ``[B, hidden_dim]``.
            candidate_item_static_features: Candidate item features ``[B, K, F2]``.

        Returns:
            Prefilter candidate scores with shape ``[B, K]``.
        """
        item_repr = self._encode_items(candidate_item_static_features)
        return torch.einsum("bd,bkd->bk", user_repr, item_repr)

    def _encode_overarch_candidates(
        self, candidate_item_static_features: torch.Tensor
    ) -> torch.Tensor:
        """Encode stage-2 candidate items with the overarch item encoder.

        Args:
            candidate_item_static_features: Candidate item features ``[B, K, F2]``.

        Returns:
            Candidate embeddings with shape ``[B, K, head_dim]``.
        """
        return self.overarch._encode_items(candidate_item_static_features)

    def _compute_overarch(
        self,
        user_sequence_features: torch.Tensor,
        user_static_features: torch.Tensor,
        candidate_item_static_features: torch.Tensor,
        prefilter_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run overarch reranking features and logits for stage-2 candidates.

        Args:
            user_sequence_features: Sequence token ids with shape ``[B, L]``.
            user_static_features: Static user features with shape ``[B, F1]``.
            candidate_item_static_features: Candidate item features ``[B, K, F2]``.
            prefilter_scores: Stage-1 scores with shape ``[B, K]``.

        Returns:
            Dictionary with:
            - ``logits``: Stage-2 logits ``[B, K]``.
            - ``u_u_dots``: User-user head interactions ``[B, H]``.
            - ``u_i_dots``: User-item head interactions ``[B, H, K]``.
            - ``prefilter_scores``: Copied stage-1 scores ``[B, K]``.
        """
        b, k, _ = candidate_item_static_features.shape
        user_heads = self.overarch._encode_user(
            user_sequence_features, user_static_features
        )
        # user_heads: [B, H, head_dim]
        candidate_overarch_repr = self._encode_overarch_candidates(
            candidate_item_static_features
        )
        # candidate_overarch_repr: [B, K, head_dim]
        # Overarch-only user-user feature: dot each head with the overarch user centroid.
        user_anchor = user_heads.mean(dim=1)
        # user_anchor: [B, head_dim]
        u_u_dots = torch.einsum("bhd,bd->bh", user_heads, user_anchor)
        # u_u_dots: [B, H]
        u_i_dots = torch.einsum(
            "bhd,bkd->bhk", user_heads, candidate_overarch_repr
        )
        # u_i_dots: [B, H, K]
        u_u_features = u_u_dots.unsqueeze(1).expand(-1, k, -1)
        u_i_features = u_i_dots.permute(0, 2, 1)
        # both tensors: [B, K, H]
        mlp_input = torch.cat(
            [prefilter_scores.unsqueeze(-1), u_u_features, u_i_features], dim=-1
        )
        # mlp_input: [B, K, 1 + 2*H]
        logits = self.overarch_mlp(mlp_input).squeeze(-1)
        # logits: [B, K]
        return {
            "logits": logits,
            "u_u_dots": u_u_dots,
            "u_i_dots": u_i_dots,
            "prefilter_scores": prefilter_scores,
        }

    def train_forward(
        self,
        user_sequence_features: torch.Tensor,
        user_static_features: torch.Tensor,
        item_static_features: torch.Tensor,
        reward_weights: Optional[torch.Tensor] = None,
        cluster_ids: Optional[torch.Tensor] = None,
        semantic_ids: Optional[torch.Tensor] = None,
        candidate_item_static_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute pointwise stage-2 BCE loss for pre-selected candidates.

        Args:
            user_sequence_features: Sequence token ids with shape ``[B, L]``.
            user_static_features: Static user features with shape ``[B, F1]``.
            item_static_features: Either unused placeholder ``[B, F2]`` or
                candidate tensor ``[B, K, F2]`` when
                ``candidate_item_static_features`` is not provided.
            reward_weights: Optional labels/weights tensor with shape ``[B, K]``.
            cluster_ids: Unused in this class.
            semantic_ids: Unused in this class.
            candidate_item_static_features: Optional explicit candidate features
                with shape ``[B, K, F2]``.

        Returns:
            Tuple ``(loss, metrics)`` where ``loss`` is scalar and ``metrics`` contains
            reranker intermediates such as logits and dot-product features.
        """
        del cluster_ids, semantic_ids
        if candidate_item_static_features is None:
            if item_static_features.dim() != 3:
                raise ValueError(
                    "MultiStageRetrieval expects candidate_item_static_features=[B,K,F2] "
                    "or item_static_features already shaped [B,K,F2]."
                )
            candidate_item_static_features = item_static_features
        prefilter_user = self._encode_user(
            user_sequence_features, user_static_features
        )
        # prefilter_user: [B, hidden_dim]
        prefilter_scores = self._prefilter_scores(
            prefilter_user, candidate_item_static_features
        )
        overarch = self._compute_overarch(
            user_sequence_features,
            user_static_features,
            candidate_item_static_features,
            prefilter_scores,
        )
        logits = overarch["logits"]
        if reward_weights is None:
            reward_weights = torch.zeros_like(logits)
            reward_weights[:, 0] = 1.0
        labels = (reward_weights > 0).float()
        losses = self.bce_loss(logits, labels)
        valid_mask = reward_weights > 0
        valid_weights = reward_weights[valid_mask]
        total_weight = valid_weights.sum()
        if total_weight < 1:
            loss = losses.sum() * 0.0
        else:
            loss = (losses[valid_mask] * valid_weights).sum() / total_weight
        return loss, {**overarch, "labels": labels}

    def inference(
        self,
        user_sequence_features: torch.Tensor,
        user_static_features: torch.Tensor,
        topk: int = 32,
    ) -> Dict[str, torch.Tensor]:
        """Run two-stage inference: prefilter retrieval then overarch reranking.

        Args:
            user_sequence_features: Sequence token ids with shape ``[B, L]``.
            user_static_features: Static user features with shape ``[B, F1]``.
            topk: Number of stage-1 candidates to rerank.

        Returns:
            Dictionary containing stage-1 outputs, stage-2 logits, and optional ids.
        """
        if self.item_pool_static is None:
            raise RuntimeError(
                "Call `build_item_pool` before running inference."
            )
        if self.item_pool_ids is None:
            raise RuntimeError(
                "Item ids were not loaded. Rebuild pool with item_ids."
            )
        prefilter_result = self.retrieve(
            user_sequence_features,
            user_static_features,
            topk=topk,
            return_all_scores=True,
        )
        scores = prefilter_result["scores"]
        indices = prefilter_result["indices"]
        flat_idx = indices.reshape(-1)
        candidate_statics = self.item_pool_static[flat_idx].view(
            indices.size(0), indices.size(1), -1
        )
        overarch = self._compute_overarch(
            user_sequence_features,
            user_static_features,
            candidate_statics,
            scores,
        )
        result: Dict[str, torch.Tensor] = {
            "prefilter": prefilter_result,
            "overarch_scores": overarch["logits"],
            "candidate_indices": indices,
        }
        result["candidate_ids"] = self.item_pool_ids[flat_idx].view(
            indices.size(0), indices.size(1)
        )
        result.update(overarch)
        return result
