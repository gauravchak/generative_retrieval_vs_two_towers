import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import CausalTransformer, MLPEncoder

class UnifiedRecEncoder(nn.Module):
    """
    Unified Recommendation Encoder.
    Combines Generative Retrieval (GR) and Two-Tower (TT) paradigms.
    A complete merge of both the structural pathways and the losses.
    """
    def __init__(self, user_input_dim, d_model, vocab_size, num_items):
        super().__init__()
        # --- PATHWAY 1: The GR Backbone ---
        self.semantic_trans = CausalTransformer(d_model) 
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # --- PATHWAY 2: The TT Backbone ---
        self.tt_user_backbone = MLPEncoder(input_dim=user_input_dim, output_dim=d_model)
        self.item_tower_embs = nn.Embedding(num_items, d_model)

        # --- FUSION: Merging TT and GR user embeddings ---
        self.fusion_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, semantic_history, user_features, target_action_enc, target_tokens, target_item_ids):
        # -------------------------------------------------------------
        # THE UNIFIED ENGINEER: "I use all inputs and combine the pathways."
        # We rely on both semantic_history/tokens and user_features/item_ids.
        # -------------------------------------------------------------
        B = semantic_history.size(0)

        # --- 1. Generative Retrieval Pathway (Using Semantic sequence) ---
        history_enc = self.semantic_trans(semantic_history)
        
        # Action-Aware Encoding for the GR side
        combined_ctx = torch.cat([history_enc, target_action_enc.unsqueeze(1)], dim=1)
        action_aware_repr = self.semantic_trans.apply_final_attn(combined_ctx)[:, -1, :] # [B, d_model]

        # --- 2. Two-Tower Pathway (Using Dense Features) ---
        tt_enc = self.tt_user_backbone(user_features) # [B, d_model]
        
        # --- 3. THE MERGE ---
        # Combine sequential semantic intent via GR with structured user profile via TT
        fused_user_vector = self.fusion_head(torch.cat([action_aware_repr, tt_enc], dim=-1))

        # --- 4. JOINT SUPERVISION (Loss Calculation) ---
        
        # A) Explicit GR Loss (using GR logic)
        loss_gr = 0
        num_target_tokens = target_tokens.size(1)
        for i in range(num_target_tokens):
            ctx = self.semantic_trans.decode_step(history_enc, target_tokens[:, :i])
            logits = self.lm_head(ctx)
            loss_gr += F.cross_entropy(logits, target_tokens[:, i])

        # B) Explicit TT Loss (using merged representation + traditional ID logic)
        item_embs = self.item_tower_embs(target_item_ids) 
        # Contrastive with in-batch negatives
        tt_logits = torch.matmul(fused_user_vector, item_embs.T)
        labels = torch.arange(B, device=tt_logits.device)
        loss_tt = F.cross_entropy(tt_logits, labels)

        # Unified objective directly merges both the causal modeling and the contrastive modeling
        return loss_gr + loss_tt
