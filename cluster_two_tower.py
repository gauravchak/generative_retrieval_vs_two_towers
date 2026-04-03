from typing import Optional

import torch
import torch.nn as nn

from two_tower_basic import TwoTowerBasic


class ClusterSoftmaxTowTower(TwoTowerBasic):
    """Two-tower with the cluster softmax extension from arXiv:2509.03746v1.

    The cluster id is treated as an extra item-side feature (`cluster_ids: [B]`) and
    supervised with a full-softmax loss over the cluster vocabulary (≈100k). This
    encourages the user tower to learn cluster-level signals quickly even when the
    item softmax still relies on sampled negatives.
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
        cluster_vocab_size: int = 100_000,
        cluster_loss_weight: float = 1.0,
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
        self.cluster_loss_weight = cluster_loss_weight
        self.cluster_embeddings = nn.Parameter(
            torch.randn(cluster_vocab_size, hidden_dim) * (hidden_dim ** -0.5)
        )
        self.cluster_bias = nn.Parameter(torch.zeros(cluster_vocab_size))
        self.cluster_loss_fn = nn.CrossEntropyLoss(reduction="none")

    def train_forward(
        self,
        user_sequence_features: torch.Tensor,
        user_static_features: torch.Tensor,
        item_static_features: torch.Tensor,
        cluster_ids: torch.Tensor,
        reward_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the usual sample softmax loss plus the cluster full softmax."""
        device = user_static_features.device
        user_repr = self._encode_user(user_sequence_features, user_static_features)
        positive_item = self._encode_items(item_static_features)

        negatives = self.oob_sampler.sample_negatives(item_static_features)
        neg_flat = negatives.view(-1, item_static_features.shape[-1])
        negative_repr = self._encode_items(neg_flat)
        negative_repr = negative_repr.view(
            item_static_features.size(0), negatives.size(1), -1
        )

        stacked_items = torch.cat([positive_item.unsqueeze(1), negative_repr], dim=1)
        logits = torch.sum(user_repr.unsqueeze(1) * stacked_items, dim=-1)

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
        losses = self.loss_fn(logits, labels)

        if reward_weights is not None:
            weights = reward_weights.to(device)
            item_loss = (losses * weights).sum() / weights.sum().clamp_min(1e-6)
        else:
            item_loss = losses.mean()

        cluster_logits = user_repr @ self.cluster_embeddings.t() + self.cluster_bias
        cluster_losses = self.cluster_loss_fn(cluster_logits, cluster_ids.to(device))
        if reward_weights is not None:
            cluster_loss = (cluster_losses * weights).sum() / weights.sum().clamp_min(1e-6)
        else:
            cluster_loss = cluster_losses.mean()

        return item_loss + self.cluster_loss_weight * cluster_loss
