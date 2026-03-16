import torch
import torch.nn as nn

class CausalTransformer(nn.Module):
    """
    A simple semantic transformer for the Generative Retrieval (GR) pathway.
    In practice, this would have multiple causal self-attention layers to process 
    sequences of semantic IDs autoregressively.
    """
    def __init__(self, d_model):
        super().__init__()
        # Simplified for educational purposes
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # x: [B, SeqLen, D]
        # In a real model, we apply a causal mask here
        attn_out, _ = self.attn(x, x, x)
        out = x + attn_out
        out = out + self.ffn(out)
        return out

    def apply_final_attn(self, x):
        """
        Applies one more attention block. This is typically used to incorporate 
        a final trigger like the 'target action' to produce the action-aware output.
        """
        return self.forward(x)

    def decode_step(self, context, prev_target_tokens):
        """
        A placeholder for autoregressive decoding. 
        It conditionally processes context and the already generated tokens.
        """
        # Returns a tensor of shape [B, D] representing the hidden state 
        # for predicting the next token
        B, _, D = context.size()
        return context.mean(dim=1)


class MLPEncoder(nn.Module):
    """
    A traditional Two-Tower (TT) user encoder.
    Typically processes dense user features and flattened ID history via MLP.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, x):
        # x: [batch_size, input_dim]
        return self.net(x)
