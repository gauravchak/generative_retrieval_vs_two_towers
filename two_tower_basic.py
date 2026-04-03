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
        positives = positives.to(self.pool.device)
        for sample in positives:
            self.pool[self.write_ptr] = sample
            self.write_ptr = (self.write_ptr + 1) % self.pool_size
            self.filled = min(self.filled + 1, self.pool_size)

    def sample_negatives(self, positives: torch.Tensor) -> torch.Tensor:
        """Return ``(B, oob_negative_count, F2)`` negatives and update the memory bank.

        If the buffer is empty (first few steps) we simply repeat the positives so that
        the downstream tower still works. Once the buffer has examples, we randomly
        draw ``B * oob_negative_count`` old items following the spirit of Mixed Negative
        Sampling: mix cached hard negatives with the current positives.
        During warm-up this keeps the returned shape consistent for the encoder.
        """
        batch_size = positives.shape[0]
        self._ingest(positives.detach())

        if self.filled == 0:
            return positives.unsqueeze(1).expand(-1, self.oob_negative_count, -1)

        max_index = self.filled
        sample_count = batch_size * self.oob_negative_count
        indices = torch.randint(max_index, (sample_count,), device=self.pool.device)
        negatives = self.pool[indices]
        negatives = negatives.to(positives.device)
        return negatives.view(batch_size, self.oob_negative_count, self.feature_dim)


class TwoTowerBasic(nn.Module):
    """A minimal two-tower retriever meant for teaching."""

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
        # item_static_features: [B, F2]
        return self.item_encoder(item_static_features)

    def build_item_index(
        self,
        item_static_features: torch.Tensor,
        item_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Encode a pool of item statics and keep them in GPU memory for inference.

        ``item_static_features`` is ``(P, F2)`` where P is the pool size of items and
        the stored embeddings are ``(P, hidden_dim)``.
        """
        embeddings = self._encode_items(item_static_features)
        self.cached_item_embeddings = embeddings.detach()
        self.cached_item_ids = item_ids.clone() if item_ids is not None else None

    def extend_item_index(
        self,
        item_static_features: torch.Tensor,
        item_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Append more items to the cached index without recomputing the whole pool."""
        embeddings = self._encode_items(item_static_features).detach()
        if self.cached_item_embeddings is None:
            self.cached_item_embeddings = embeddings
        else:
            self.cached_item_embeddings = torch.cat(
                [self.cached_item_embeddings, embeddings], dim=0
            )
        if item_ids is not None:
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
        """Run a simple matmul between user repr and cached item embeddings for inference.

        This relies on the cached item embeddings to stay resident in GPU memory. The
        returned dictionary always contains ``scores`` and ``indices`` and optionally
        ``item_ids`` / ``logits`` for diagnostics.
        """
        if self.cached_item_embeddings is None:
            raise RuntimeError("Call `build_item_index` before retrieval.")
        user_repr = self._encode_user(user_sequence_features, user_static_features)
        # logits: [B, num_cached_items]
        logits = user_repr @ self.cached_item_embeddings.t()
        max_k = min(topk, logits.size(1))
        scores, indices = logits.topk(max_k, dim=-1)
        # scores: [B, topk], indices: [B, topk]
        result: Dict[str, torch.Tensor] = {"scores": scores, "indices": indices}
        if self.cached_item_ids is not None:
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
    ) -> torch.Tensor:
        """Compute logits for positives and out-of-batch negatives, return a weighted loss."""
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
        # stacked_items: [B, 1 + num_oob, hidden_dim]
        logits = torch.sum(user_repr.unsqueeze(1) * stacked_items, dim=-1)
        # logits: [B, 1 + num_oob]

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
        losses = self.loss_fn(logits, labels)
        if reward_weights is not None:
            losses = losses * reward_weights.to(device)
            denominator = reward_weights.to(device).sum().clamp_min(1e-6)
            loss = losses.sum() / denominator
        else:
            loss = losses.mean()
        return loss
