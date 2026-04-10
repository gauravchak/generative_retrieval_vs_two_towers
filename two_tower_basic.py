from typing import Dict, Optional

import torch
import torch.nn as nn


class OOBNegativeSampler:
    """Out-Of-Batch (OOB) sampler inspired by Google's Mixed Negative Sampling paper.

    The sampler holds a small memory bank of historical item representations and randomly
    mixes them into the current batch so that each positive example sees ``oob_negative_count``
    stale negatives in addition to its in-batch logits. The sampler operates on the static
    item-side features that the tower will encode, rather than logits directly.
    Positives are ``(B, F2)`` tensors and the sampler returns ``(B, oob_negative_count, F2)``.
    """

    def __init__(
        self,
        feature_dim: int,
        pool_size: int = 4096,
        oob_negative_count: int = 4,
    ) -> None:
        self.feature_dim = feature_dim
        self.pool_size = pool_size
        self.oob_negative_count = oob_negative_count
        self.pool = torch.zeros(pool_size, feature_dim)
        self.write_ptr = 0
        self.filled = 0

    def _ingest(self, positives: torch.Tensor) -> None:
        """Insert current positives into the rolling negative pool.

        Args:
            positives: Item-side feature tensor with shape ``[B, F2]``.

        Returns:
            None. Updates internal pool state in place.
        """
        positives = positives.to(self.pool.device)
        for sample in positives:
            self.pool[self.write_ptr] = sample
            self.write_ptr = (self.write_ptr + 1) % self.pool_size
            self.filled = min(self.filled + 1, self.pool_size)

    def get_candidates(self, positives: torch.Tensor) -> torch.Tensor:
        """Sample out-of-batch negatives and prepend the positive items.

        Args:
            positives: Current positive item features with shape ``[B, F2]``.

        Returns:
            Candidates with shape ``[B, 1 + oob_negative_count, F2]`` where index 0 is positive.
        """
        batch_size = positives.shape[0]
        self._ingest(positives.detach())

        if self.filled == 0:
            negatives = positives.unsqueeze(1).expand(
                -1, self.oob_negative_count, -1
            )
        else:
            max_index = self.filled
            sample_count = batch_size * self.oob_negative_count
            indices = torch.randint(
                max_index, (sample_count,), device=self.pool.device
            )
            negatives = self.pool[indices]
            negatives = negatives.to(positives.device).view(
                batch_size, self.oob_negative_count, self.feature_dim
            )

        return torch.cat([positives.unsqueeze(1), negatives], dim=1)


class TwoTowerBasic(nn.Module):
    """A minimal two-tower retriever meant for teaching.

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
        oob_pool_size: int = 4096,
        oob_negative_count: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.user_sequence_length = user_sequence_length

        self.user_embedding = nn.Embedding(user_vocab_size, user_embedding_dim)
        self.user_sequence_projection = nn.Sequential(
            nn.Linear(user_sequence_length * user_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.user_static_projection = nn.Sequential(
            nn.Linear(user_static_dim, hidden_dim),
            nn.ReLU(),
        )
        # Final projection fuses sequence context with static context into D dimensions.
        self.user_final = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        self.item_encoder = nn.Sequential(
            nn.Linear(item_static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.oob_sampler = OOBNegativeSampler(
            feature_dim=item_static_dim,
            pool_size=oob_pool_size,
            oob_negative_count=oob_negative_count,
        )
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.cached_item_embeddings: Optional[torch.Tensor] = None
        self.cached_item_ids: Optional[torch.Tensor] = None

    def _encode_user(
        self,
        user_sequence_features: torch.Tensor,
        user_static_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode user sequence and static features into a single vector.

        Args:
            user_sequence_features: Sequence token ids with shape ``[B, L]``.
            user_static_features: Static user features with shape ``[B, F1]``.

        Returns:
            User embedding tensor with shape ``[B, hidden_dim]``.
        """
        # user_sequence_features: [B, L] -> embeddings [B, L, D]
        embeds = self.user_embedding(user_sequence_features)
        seq_flat = embeds.reshape(embeds.size(0), -1)
        # seq_flat: [B, L*D]
        seq_repr = self.user_sequence_projection(seq_flat)
        # seq_repr: [B, hidden_dim]
        static_repr = self.user_static_projection(user_static_features)
        # static_repr: [B, hidden_dim]
        user_repr = self.user_final(torch.cat([seq_repr, static_repr], dim=-1))
        # user_repr: [B, hidden_dim]
        return user_repr

    def _encode_items(self, item_static_features: torch.Tensor) -> torch.Tensor:
        """Encode item-side static features into the retrieval space.

        Args:
            item_static_features: Item features with shape ``[..., F2]``.

        Returns:
            Item embeddings with shape ``[..., hidden_dim]``.
        """
        return self.item_encoder(item_static_features)

    def build_item_index(
        self,
        item_static_features: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> None:
        """Build a cached embedding index for inference-time retrieval.

        Args:
            item_static_features: Item pool features with shape ``[P, F2]``.
            item_ids: Integer ids aligned to ``item_static_features``, shape ``[P]``.

        Returns:
            None. Stores cached embeddings and ids on the model instance.
        """
        embeddings = self._encode_items(item_static_features)
        self.cached_item_embeddings = embeddings.detach()
        self.cached_item_ids = item_ids.clone()

    def extend_item_index(
        self,
        item_static_features: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> None:
        """Append additional items to the cached index.

        Args:
            item_static_features: New item features with shape ``[P_new, F2]``.
            item_ids: Ids for the new items with shape ``[P_new]``.

        Returns:
            None. Extends cached embeddings and ids in place.
        """
        embeddings = self._encode_items(item_static_features).detach()
        if self.cached_item_embeddings is None:
            self.cached_item_embeddings = embeddings
        else:
            self.cached_item_embeddings = torch.cat(
                [self.cached_item_embeddings, embeddings], dim=0
            )
        if self.cached_item_ids is None:
            self.cached_item_ids = item_ids.clone()
        else:
            self.cached_item_ids = torch.cat(
                [self.cached_item_ids, item_ids.clone()], dim=0
            )

    def retrieve(
        self,
        user_sequence_features: torch.Tensor,
        user_static_features: torch.Tensor,
        topk: int = 10,
        return_all_scores: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run top-k retrieval by matmul against cached item embeddings.

        Args:
            user_sequence_features: Sequence token ids with shape ``[B, L]``.
            user_static_features: Static user features with shape ``[B, F1]``.
            topk: Number of top candidates to return.
            return_all_scores: If ``True``, include full logits ``[B, P]`` in output.

        Returns:
            Dictionary containing:
            - ``scores``: Top-k scores, shape ``[B, topk]``.
            - ``indices``: Top-k indices into cached item pool, shape ``[B, topk]``.
            - ``item_ids``: Top-k business ids with shape ``[B, topk]``.
            - ``logits`` (optional): Full score matrix when requested.
        """
        if self.cached_item_embeddings is None:
            raise RuntimeError("Call `build_item_index` before retrieval.")
        if self.cached_item_ids is None:
            raise RuntimeError(
                "Item ids were not loaded. Rebuild index with item_ids."
            )
        user_repr = self._encode_user(
            user_sequence_features, user_static_features
        )
        # logits: [B, num_cached_items]
        logits = user_repr @ self.cached_item_embeddings.t()
        max_k = min(topk, logits.size(1))
        scores, indices = logits.topk(max_k, dim=-1)
        # scores: [B, topk], indices: [B, topk]
        result: Dict[str, torch.Tensor] = {"scores": scores, "indices": indices}
        result["item_ids"] = self.cached_item_ids[indices]
        if return_all_scores:
            result["logits"] = logits
        return result

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
        """Compute sampled-softmax retrieval loss for the two-tower objective.

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
        user_repr = self._encode_user(
            user_sequence_features, user_static_features
        )

        candidates = self.oob_sampler.get_candidates(item_static_features)
        candidate_repr = self._encode_items(candidates)
        # candidate_repr: [B, 1 + num_oob, hidden_dim]

        logits = torch.einsum("bd,bkd->bk", user_repr, candidate_repr)
        # logits: [B, 1 + num_oob]

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
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
