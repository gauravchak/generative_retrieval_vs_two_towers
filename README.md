# A Unified Retrieval Argument

## Core Claim
This repository argues that generative retrieval and two-tower retrieval should not be treated as separate, competing systems.

Instead, we can train them together in one model family, reuse the same user/item representations, and choose the inference path based on product constraints:
- Fast embedding retrieval with matrix multiplication (tower path)
- Autoregressive semantic-ID generation (generative path)

If this is done correctly, the unified system should be at least as good as current two-tower state of the art for retrieval quality and latency-sensitive serving, while also unlocking generative retrieval behaviors without requiring a separate model stack.

## Step 1: Build a Strong Two-Tower Baseline (`TwoTowerBasic`)
We start from the known strong baseline and make it educational:
- User encoder combines sequence history and lightweight static features.
- Item encoder produces item embeddings for dot-product retrieval.
- OOB negative sampler (mixed-negative style) increases training signal beyond in-batch negatives.
- Inference uses a cached GPU item embedding table and direct matmul for top-k retrieval.

Why this step matters:
- This baseline already reflects how production two-tower systems are commonly trained and served.
- It gives us a stable anchor to compare all later variants.

## Step 2: Add Late Interaction (`MultiHeadTwoTower`)
Next we add a ColBERT-style variant with minimal change:
- Replace single user embedding with multiple user heads.
- Keep item encoding and negative-sampling workflow familiar.
- Aggregate head-wise similarities (max/logsumexp).

Why this step matters:
- It shows richer interaction can be layered on top of the same two-tower foundation.
- It keeps compatibility with the same data and training loop style.

```mermaid
flowchart LR
    U["User tower"] --> H1["Head 1: u1"]
    U --> H2["Head 2: u2"]
    U --> H3["Head 3: u3"]
    U --> H4["Head 4: u4"]
    I["Item tower: i"] --> S1["u1 dot i"]
    I --> S2["u2 dot i"]
    I --> S3["u3 dot i"]
    I --> S4["u4 dot i"]
    H1 --> S1
    H2 --> S2
    H3 --> S3
    H4 --> S4
    S1 --> A["Aggregate max/logsumexp"]
    S2 --> A
    S3 --> A
    S4 --> A
    A --> L["Final item score"]
```

## Step 3: Add Two-Stage Retrieval (`MultiStageRetrieval`)
Now we extend to a prefilter + overarch setup:
- Stage 1 prefilter: fast KNN-style matmul over cached embeddings.
- Stage 2 overarch: produce a separate overarch user representation (`[B, H, D]`) and
  compute `u_u_dots` and `u_i_dots` from that overarch branch; combine these with
  stage-1 prefilter scores in an MLP.
- Train with impression-style pointwise objectives.
- Reference: [Revisiting Neural Retrieval on Accelerators](https://arxiv.org/abs/2306.04039).

Why this step matters:
- It preserves serving efficiency while improving ranking expressiveness.
- It demonstrates that multi-stage retrieval is still compatible with the same base architecture.

```mermaid
flowchart LR
    Q["User features"] --> P["Prefilter user embedding"]
    P --> KNN["GPU matmul KNN"]
    KNN --> TOPK["Top K candidates"]
    TOPK --> OI["Overarch item encoding"]
    Q --> OU["Overarch multi head user embedding"]
    OU --> UUD["UU dots from overarch user branch"]
    OU --> UID["UI dots"]
    OI --> UID
    TOPK --> PS["Prefilter scores"]
    UID --> MLP["Overarch MLP"]
    UUD --> MLP
    PS --> MLP
    MLP --> R["Re ranked scores"]
```

## Step 4: Add Cluster Supervision (`ClusterSoftmaxTowTower`)
We then add a full-softmax cluster objective on item-side cluster IDs:
- Keep sampled-softmax item objective.
- Add full-softmax cluster loss (hashed large cluster vocabulary).
- Jointly optimize both.

Why this step matters:
- Cluster targets are often easier to learn early.
- Full-softmax cluster supervision quickly corrects coarse semantic mistakes.
- This improves learning dynamics without discarding two-tower strengths.

## Step 5: Add Generative Retrieval (`GenerativeRetrieval`)
Now we add semantic-ID generation with teacher forcing:
- User side stays intentionally light (history tokens + static tokens concatenated on token axis).
- Decoder follows OneRecV-style block ordering:
  1. cross-attention to user tokens
  2. self-attention on generated tokens
  3. MoE-FFN
- Target is semantic token sequence (for example length 4 with BOS).

Why this step matters:
- It captures TIGER-like generative retrieval behavior.
- It adds a new end-to-end generative inference path for item prediction, not just
  KNN + overarch-style inference.
- It does not require abandoning the existing two-tower ecosystem.

```mermaid
flowchart LR
    U["User tokens memory"] --> DT["Decoder (training lane)"]
    U --> DI["Decoder (inference lane)"]

    GT["BOS + gold token t-1"] --> DT
    DT --> PT["Predict token t"]
    PT --> LT["CE loss vs gold token t"]

    GI["BOS + generated token t-1"] --> DI
    DI --> PI["Predict token t"]
    PI --> FB["Append token and continue"]
```

## Step 6: Close the Loop with Joint Training (`UnifiedRetrieval`)
Finally, we combine both objectives in one model:
- Tower sampled-softmax loss
- Generative semantic-token loss
- Weighted sum during training

At inference, we expose two callable paths from the same trained model:
- `retrieve_with_tower(...)` for fast ANN/matmul style retrieval
- `generate_semantic_ids(...)` for generative retrieval

Why this is the critical result:
- We are no longer forced into an either/or choice between discriminative and generative retrieval.
- We can keep the proven two-tower serving path and quality, while adding generative capability as a first-class option.

```mermaid
flowchart TD
    TRAIN["Shared<br/>training batch"] --> TL["Tower<br/>loss"]
    TRAIN --> GL["Generation<br/>loss"]
    TL --> SUM["Weighted<br/>sum"]
    GL --> SUM
    SUM --> M["One unified<br/>checkpoint"]
    M --> INF1["Inference A<br/>retrieve_with_tower"]
    M --> INF2["Inference B<br/>generate_semantic_ids"]
```

## Why This Makes Sense In Practice
If you are already running a strong two-tower stack, this approach is a low-risk extension, not a rewrite.

Why the logic is sound:
- You keep the proven two-tower objective and serving path.
- You add a second objective (semantic generation) on top of the same user/item training examples.
- You get one checkpoint that supports two inference modes, so you can pick by latency/product need.

Why this should be convincing to a skeptical reader:
- It is additive: no need to throw away existing two-tower infrastructure.
- It is testable: you can compare tower-only, generative-only, and joint training from the same data backbone.
- It is operationally flexible: fast matmul retrieval remains available even after adding generation.

## What You Need To Try This
Minimum data requirements:
- The same interaction/impression logs you already use for two-tower training.
- Item-side features used by your current tower model.
- A semantic id sequence per item (for example 4 tokens with BOS), logged or precomputed and joined by item id.

Minimum training setup:
- Keep your current two-tower loss.
- Add teacher-forcing semantic-token loss.
- Train jointly with configurable loss weights.
- Track both retrieval metrics (for tower path) and token/generative metrics (for generation path).

Minimum serving setup:
- Keep the existing GPU index + matmul path (`retrieve_with_tower`).
- Add semantic autoregressive decoding path (`generate_semantic_ids`).
- Route traffic by use case, or run both in parallel for evaluation.

Bottom line:
- Unified retrieval is a practical superset strategy: at least preserve current two-tower strength, while gaining a production-ready path to end-to-end generative retrieval.

## Evolution Diagram
```mermaid
flowchart TD
    A[TwoTowerBasic<br/>strong baseline + OOB + matmul serving]
    B[MultiHeadTwoTower<br/>late interaction with multiple heads]
    C[MultiStageRetrieval<br/>prefilter + overarch reranker]
    D[ClusterSoftmaxTowTower<br/>item loss + cluster full softmax]
    E[GenerativeRetrieval<br/>token decoding with OneRecV-style blocks]
    F[UnifiedRetrieval<br/>joint losses + dual inference paths]

    A --> B
    A --> C
    A --> D
    A --> E
    B --> F
    C --> F
    D --> F
    E --> F
```
