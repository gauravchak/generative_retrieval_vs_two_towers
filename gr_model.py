import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import CausalTransformer

class GenerativeRetrievalModel(nn.Module):
    """
    Standalone Generative Retrieval (GR) Model.
    A GR engineer will recognize this: autoregressively generating the item's semantic IDs.
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        # GR Backbone to encode sequences
        self.semantic_trans = CausalTransformer(d_model)
        
        # LM Head mapping representations to semantic token vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, semantic_history, user_features, target_action_enc, target_tokens, target_item_ids):
        # -------------------------------------------------------------
        # GR ENGINEER PERSPECTIVE: "I only care about the semantic sequence."
        # Ignored inputs for pure GR:
        # - user_features (No dense features)
        # - target_item_ids (No traditional item catalog ID logic)
        # -------------------------------------------------------------
        
        B = semantic_history.size(0)

        # 1. Encode semantic history via transformer
        history_enc = self.semantic_trans(semantic_history)

        # 2. Action-Aware Encoding (Optional for GR, but useful for trigger contexts)
        combined_ctx = torch.cat([history_enc, target_action_enc.unsqueeze(1)], dim=1)
        action_aware_repr = self.semantic_trans.apply_final_attn(combined_ctx)

        # 3. Calculate language modeling loss (Teacher Forcing)
        loss_gr = 0
        num_target_tokens = target_tokens.size(1)
        
        for i in range(num_target_tokens):
            # Predict the next token based on history + previous target tokens.
            ctx = self.semantic_trans.decode_step(history_enc, target_tokens[:, :i])
            
            # Project to vocabulary probabilities
            logits = self.lm_head(ctx)
            
            # Cross entropy against the i-th true token
            loss_gr += F.cross_entropy(logits, target_tokens[:, i])

        return loss_gr
