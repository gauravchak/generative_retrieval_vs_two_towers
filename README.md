# Generative Retrieval vs Two Towers

This repository provides an educational, schematic comparison between two distinct paradigms in modern Recommendation Systems: **Generative Retrieval (GR)** and **Two-Tower (TT)** retrieval. Finally, it demonstrates a **Unified** approach that merges both paradigms.

## The Concept

The core idea is to show how different engineers might approach the same recommendation problem using different inputs and loss functions:

*   **Two-Tower (TT) Engineer**: Cares about dense user features and item catalogs. They build a User Tower and an Item Tower, optimizing via contrastive learning (e.g., in-batch negatives).
*   **Generative Retrieval (GR) Engineer**: Cares about the sequence of user actions as semantic tokens. They build an autoregressive model to predict the next semantic ID in the sequence (using language modeling/teacher forcing).
*   **Unified Engineer**: Combines both. They fuse the sequence-aware context from the GR pathway with the dense feature representation from the TT pathway, optimizing a joint loss function.

## Code Structure

To illustrate this, all models share the exact same `forward` signature to highlight which inputs each paradigm actually uses (and ignores).

*   `modules.py`: Contains the base network components like `CausalTransformer` and `MLPEncoder`.
*   `tt_model.py`: Standalone Two-Tower implementation. It ignores all semantic history and focuses purely on mapping user features to item IDs.
*   `gr_model.py`: Standalone Generative Retrieval implementation. It ignores dense user features and item catalogs, focusing solely on autoregressively predicting target semantic tokens.
*   `unified_model.py`: The merged architecture that combines the outputs of both pathways and computes a joint `loss_gr + loss_tt`.

## Note
This code is highly schematic and intended for educational purposes to understand the structural and data-flow differences between these recommendation architectures.
