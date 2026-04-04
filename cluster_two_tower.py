from typing import Optional

import torch
import torch.nn as nn

from two_tower_basic import TwoTowerBasic


class ClusterSoftmaxTowTower(TwoTowerBasic):
    """Two-tower with the cluster softmax extension from arXiv:2509.03746v1.

    The cluster id is treated as an extra item-side feature (`cluster_ids: [B]`) and
    supervised with a full-softmax loss over the cluster vocabulary (≈100k). This helps
    the model separate the right L1 cluster from incorrect clusters early in training.
    For very large catalogs this remains computationally feasible because practical
    cluster counts are around ``sqrt(|I|)``; for ``|I|≈5B`` this is about ``70k``.

    For inference we optionally index ``item_embedding + cluster_embedding`` in the
    ANN/matmul cache, giving:
    ``<u, item+cluster> = <u, item> + <u, cluster>``.
    This makes it easier to down-rank items from unlikely clusters.

    Unified ``train_forward`` API:
    - used: ``user_sequence_features``, ``user_static_features``, ``item_static_features``,
      ``cluster_ids``, optional ``reward_weights``
    - ignored: ``semantic_ids``, ``candidate_item_static_features``
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

    def build_item_index(
        self,
        item_static_features: torch.Tensor,
        item_ids: torch.Tensor,
        cluster_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Build retrieval cache, optionally adding cluster embeddings to items.

        Args:
            item_static_features: Item features with shape ``[P, F2]``.
            item_ids: Ids aligned to ``item_static_features``, shape ``[P]``.
            cluster_ids: Optional cluster ids aligned to ``item_static_features``,
                shape ``[P]``. If provided, cache ``item + cluster`` embeddings.

        Returns:
            None. Updates cached item embeddings used by retrieval.
        """
        embeddings = self._encode_items(item_static_features).detach()
        if cluster_ids is not None:
            cluster_vecs = self.cluster_embeddings[cluster_ids.to(embeddings.device)].detach()
            embeddings = embeddings + cluster_vecs
        self.cached_item_embeddings = embeddings
        self.cached_item_ids = item_ids.clone()

    def extend_item_index(
        self,
        item_static_features: torch.Tensor,
        item_ids: torch.Tensor,
        cluster_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Append rows to retrieval cache, with optional cluster-aware embeddings.

        Args:
            item_static_features: New item features with shape ``[P_new, F2]``.
            item_ids: Ids for new rows with shape ``[P_new]``.
            cluster_ids: Optional cluster ids for new rows with shape ``[P_new]``.

        Returns:
            None. Extends cached embeddings and optional ids.
        """
        embeddings = self._encode_items(item_static_features).detach()
        if cluster_ids is not None:
            cluster_vecs = self.cluster_embeddings[cluster_ids.to(embeddings.device)].detach()
            embeddings = embeddings + cluster_vecs
        if self.cached_item_embeddings is None:
            self.cached_item_embeddings = embeddings
        else:
            self.cached_item_embeddings = torch.cat([self.cached_item_embeddings, embeddings], dim=0)
        if self.cached_item_ids is None:
            self.cached_item_ids = item_ids.clone()
        else:
            self.cached_item_ids = torch.cat([self.cached_item_ids, item_ids.clone()], dim=0)

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
        """Compute sampled-softmax item loss plus full-softmax cluster loss.

        Args:
            user_sequence_features: Sequence token ids with shape ``[B, L]``.
            user_static_features: Static user features with shape ``[B, F1]``.
            item_static_features: Positive item features with shape ``[B, F2]``.
            reward_weights: Optional per-example weights with shape ``[B]``.
            cluster_ids: Cluster labels with shape ``[B]`` (required).
            semantic_ids: Unused in this class.
            candidate_item_static_features: Unused in this class.

        Returns:
            Scalar training loss tensor.
        """
        del semantic_ids, candidate_item_static_features
        if cluster_ids is None:
            raise ValueError("cluster_ids is required for ClusterSoftmaxTowTower.")
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

        # Cluster full-softmax quickly teaches coarse cluster separation before
        # item-level sampled-softmax has fully converged.
        cluster_logits = user_repr @ self.cluster_embeddings.t() + self.cluster_bias
        cluster_losses = self.cluster_loss_fn(cluster_logits, cluster_ids.to(device))
        if reward_weights is not None:
            cluster_loss = (cluster_losses * weights).sum() / weights.sum().clamp_min(1e-6)
        else:
            cluster_loss = cluster_losses.mean()

        return item_loss + self.cluster_loss_weight * cluster_loss
