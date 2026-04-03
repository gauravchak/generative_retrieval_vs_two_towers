from typing import Dict, Optional

import torch
import torch.nn as nn

from multi_head_two_tower import MultiHeadTwoTower
from two_tower_basic import TwoTowerBasic


class MultiStageRetrieval(TwoTowerBasic):
    """Two-stage retriever (arXiv:2306.04039) that builds directly on ``TwoTowerBasic``.

    By subclassing ``TwoTowerBasic`` we reuse the prefilter-stage encode / OOB sampler /
    cached item-index behavior and only add the overarch logic on top. That keeps the diff
    focused on the late-interaction reranker while still explaining the extra tensors.
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
        self.prefilter_user_projection = nn.Linear(hidden_dim, head_dim)
        self.overarch_mlp = nn.Sequential(
            nn.Linear(1 + 2 * head_dim, overarch_hidden),
            nn.ReLU(),
            nn.Linear(overarch_hidden, 1),
        )
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.item_pool_static: Optional[torch.Tensor] = None
        self.item_pool_ids: Optional[torch.Tensor] = None

    def build_item_pool(
        self, item_static_features: torch.Tensor, item_ids: Optional[torch.Tensor] = None
    ) -> None:
        """Repurpose ``TwoTowerBasic.build_item_index`` as the stage-1 cache.

        We also keep the raw static rows to re-encode selected candidates in stage 2.
        """
        super().build_item_index(item_static_features, item_ids)
        self.item_pool_static = item_static_features.detach()
        self.item_pool_ids = item_ids.clone() if item_ids is not None else None

    def extend_item_pool(
        self, item_static_features: torch.Tensor, item_ids: Optional[torch.Tensor] = None
    ) -> None:
        """Extend both the prefilter cache and the stored statics for stage 2 reranking."""
        super().extend_item_index(item_static_features, item_ids)
        if self.item_pool_static is None:
            self.item_pool_static = item_static_features.detach()
        else:
            self.item_pool_static = torch.cat(
                [self.item_pool_static, item_static_features.detach()], dim=0
            )
        if item_ids is not None:
            if self.item_pool_ids is None:
                self.item_pool_ids = item_ids.clone()
            else:
                self.item_pool_ids = torch.cat(
                    [self.item_pool_ids, item_ids.clone()], dim=0
                )

    def _prefilter_scores(
        self, user_repr: torch.Tensor, candidate_item_static_features: torch.Tensor
    ) -> torch.Tensor:
        """Dot product between the prefilter user vector and candidates.

        user_repr: [B, hidden_dim]
        candidate_item_static_features: [B, K, F2]
        returns: [B, K]
        """
        b, k, f2 = candidate_item_static_features.shape
        flat = candidate_item_static_features.reshape(-1, f2)
        item_repr = self._encode_items(flat).view(b, k, -1)
        return torch.sum(user_repr.unsqueeze(1) * item_repr, dim=-1)

    def _encode_overarch_candidates(self, candidate_item_static_features: torch.Tensor) -> torch.Tensor:
        """Encode rerank candidates with the multi-head item encoder."""
        b, k, f2 = candidate_item_static_features.shape
        flat = candidate_item_static_features.reshape(-1, f2)
        item_repr = self.overarch._encode_items(flat)
        return item_repr.view(b, k, -1)

    def _compute_overarch(
        self,
        user_sequence_features: torch.Tensor,
        user_static_features: torch.Tensor,
        candidate_item_static_features: torch.Tensor,
        prefilter_user: torch.Tensor,
        prefilter_scores: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run the overarch stage using the multi-head reranker."""
        b, k, _ = candidate_item_static_features.shape
        user_heads = self.overarch._encode_user(
            user_sequence_features, user_static_features
        )
        # user_heads: [B, H, head_dim]
        candidate_overarch_repr = self._encode_overarch_candidates(
            candidate_item_static_features
        )
        # candidate_overarch_repr: [B, K, head_dim]
        prefilter_user_proj = self.prefilter_user_projection(prefilter_user)
        # prefilter_user_proj: [B, head_dim]
        u_u_dots = torch.einsum("bhd,bd->bh", user_heads, prefilter_user_proj)
        # u_u_dots: [B, H]
        u_i_dots = torch.einsum("bhd,bkd->bhk", user_heads, candidate_overarch_repr)
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
        candidate_item_static_features: torch.Tensor,
        reward_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the pointwise overarch loss on pre-selected candidates."""
        prefilter_user = self._encode_user(user_sequence_features, user_static_features)
        # prefilter_user: [B, hidden_dim]
        prefilter_scores = self._prefilter_scores(
            prefilter_user, candidate_item_static_features
        )
        overarch = self._compute_overarch(
            user_sequence_features,
            user_static_features,
            candidate_item_static_features,
            prefilter_user,
            prefilter_scores,
        )
        logits = overarch["logits"]
        if reward_weights is None:
            reward_weights = torch.zeros_like(logits)
            reward_weights[:, 0] = 1.0
        labels = (reward_weights > 0).float()
        losses = self.bce_loss(logits, labels)
        loss = (losses * reward_weights).sum() / reward_weights.sum().clamp_min(1e-6)
        return loss, {**overarch, "labels": labels}

    def inference(
        self,
        user_sequence_features: torch.Tensor,
        user_static_features: torch.Tensor,
        topk: int = 32,
    ) -> Dict[str, torch.Tensor]:
        """Prefilter via matmul KNN, then rerank with the overarch MLP."""
        if self.item_pool_static is None:
            raise RuntimeError("Call `build_item_pool` before running inference.")
        prefilter_result = self.retrieve(
            user_sequence_features, user_static_features, topk=topk, return_all_scores=True
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
            self._encode_user(user_sequence_features, user_static_features),
            scores,
        )
        result: Dict[str, torch.Tensor] = {
            "prefilter": prefilter_result,
            "overarch_scores": overarch["logits"],
            "candidate_indices": indices,
        }
        if self.item_pool_ids is not None:
            result["candidate_ids"] = self.item_pool_ids[flat_idx].view(
                indices.size(0), indices.size(1)
            )
        result.update(overarch)
        return result
