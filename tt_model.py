import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import MLPEncoder

class TwoTowerModel(nn.Module):
    """
    Standalone Two-Tower (TT) Model.
    A traditional RecSys engineer will recognize this: contrastive learning between a User Tower and an Item Tower.
    """
    def __init__(self, user_input_dim, d_model, num_items):
        super().__init__()
        # User Tower encoding dense user features -> [B, d_model]
        self.user_tower = MLPEncoder(input_dim=user_input_dim, output_dim=d_model)
        
        # Item Tower (lookup table or another MLP) -> [num_items, d_model]
        self.item_tower_embs = nn.Embedding(num_items, d_model)

    def forward(self, semantic_history, user_features, target_action_enc, target_tokens, target_item_ids):
        # -------------------------------------------------------------
        # TT ENGINEER PERSPECTIVE: "I just need the dense user features and the target item ID."
        # Ignored inputs for pure TT: 
        # - semantic_history (No semantic sequence modeling)
        # - target_action_enc (No semantic action modeling)
        # - target_tokens (No generative token prediction)
        # -------------------------------------------------------------
        
        B = user_features.size(0)

        # 1. Encode user using traditional MLP
        user_enc = self.user_tower(user_features)  # [B, d_model]

        # 2. Lookup item embeddings for the positive target items
        item_embs = self.item_tower_embs(target_item_ids) # [B, d_model]
        
        # 3. Calculate logits (Dot product for contrastive learning)
        # Using in-batch negatives: dot multiplying every user against every item in batch
        logits = torch.matmul(user_enc, item_embs.T) # [B, B]
        
        # 4. TT Loss (Contrastive / In-batch Negatives)
        # The correct item for user i is item i (the diagonal)
        labels = torch.arange(B, device=logits.device)
        loss_tt = F.cross_entropy(logits, labels)

        return loss_tt
