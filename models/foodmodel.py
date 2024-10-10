import torch
import torch.nn as nn
from modules.transformer import TransformerEncoder
import torch.nn.functional as F


class FoodModel(nn.Module):
    def __init__(self, output_dim=101, num_heads=5, layers=4,
                 relu_dropout=0.1, embed_dropout=0.3, res_dropout=0.1, out_dropout=0.1,
                 attn_dropout=0.25):
        super(FoodModel, self).__init__()
        self.num_mod = 2
        self.proj_dim = 40
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout
        self.projv = nn.Conv1d(768, self.proj_dim, kernel_size=1, padding=0)
        self.projt = nn.Conv1d(768, self.proj_dim, kernel_size=1, padding=0)
        self.vision_encoder = TransformerEncoder(
            embed_dim=self.proj_dim, num_heads=self.num_heads,
            layers=self.layers, attn_dropout=self.attn_dropout, res_dropout=self.res_dropout,
            relu_dropout=self.relu_dropout, embed_dropout=self.embed_dropout
        )
        self.text_encoder = TransformerEncoder(
            embed_dim=self.proj_dim, num_heads=self.num_heads,
            layers=self.layers, attn_dropout=self.attn_dropout, res_dropout=self.res_dropout,
            relu_dropout=self.relu_dropout, embed_dropout=self.embed_dropout
        )

        self.fusion = TransformerEncoder(
            embed_dim=self.proj_dim, num_heads=self.num_heads,
            layers=self.layers-2, attn_dropout=self.attn_dropout, res_dropout=self.res_dropout,
            relu_dropout=self.relu_dropout, embed_dropout=self.embed_dropout
        )

        # Output layers
        self.proj1 = nn.Linear(self.proj_dim, self.proj_dim)
        self.proj2 = nn.Linear(self.proj_dim, self.proj_dim)
        self.out_layer = nn.Linear(self.proj_dim, output_dim)

    def forward(self, v, t):
        v = v.transpose(1, 2)
        v = self.projv(v)
        v = v.permute(2, 0, 1)
        t = t.transpose(1, 2)
        t = self.projv(t)
        t = t.permute(2, 0, 1)
        v = self.vision_encoder(v)
        t = self.text_encoder(t)
        hs = [v.clone().detach(), t.clone().detach()]
        f = torch.cat([v, t], dim=0)
        last_hs = self.fusion(f)[0]
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output, hs
