import torch
from torch import nn
import torch.nn.functional as F
from src.eval_metrics import *

from modules.transformer import TransformerEncoder


class MSAModel(nn.Module):
    def __init__(self, output_dim, orig_dim, proj_dim=30, num_heads=5, layers=5,
                 relu_dropout=0.1, embed_dropout=0.3, res_dropout=0.1, out_dropout=0.1,
                 attn_dropout=0.25
                 ):
        super(MSAModel, self).__init__()

        self.proj_dim = proj_dim
        self.orig_dim = orig_dim
        self.num_mod = len(orig_dim)
        self.num_heads = num_heads
        self.layers = layers
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.out_dropout = out_dropout
        self.embed_dropout = embed_dropout

        # Projection Layers
        self.proj = nn.ModuleList([
            nn.Conv1d(self.orig_dim[i], self.proj_dim, kernel_size=1, padding=0)
            for i in range(self.num_mod)
        ])

        # Encoders
        self.encoders = nn.ModuleList([
            TransformerEncoder(embed_dim=proj_dim, num_heads=self.num_heads,
                               layers=self.layers, attn_dropout=self.attn_dropout, res_dropout=self.res_dropout,
                               relu_dropout=self.relu_dropout, embed_dropout=self.embed_dropout)
            for _ in range(self.num_mod)
        ])

        # Fusion
        self.fusion = TransformerEncoder(
            embed_dim=proj_dim, num_heads=self.num_heads,
            layers=self.layers-2, attn_dropout=self.attn_dropout, res_dropout=self.res_dropout,
            relu_dropout=self.relu_dropout, embed_dropout=self.embed_dropout
        )

        # Output layers
        self.proj1 = nn.Linear(self.proj_dim, self.proj_dim)
        self.proj2 = nn.Linear(self.proj_dim, self.proj_dim)
        self.out_layer = nn.Linear(self.proj_dim, output_dim)

    def forward(self, x):
        """
        dimension [batch_size, seq_len, n_features]
        """
        hs = list()
        hs_detach = list()

        for i in range(self.num_mod):
            x[i] = x[i].transpose(1, 2)
            x[i] = self.proj[i](x[i])
            x[i] = x[i].permute(2, 0, 1)
            h_tmp = self.encoders[i](x[i])
            hs.append(h_tmp)
            hs_detach.append(h_tmp.clone().detach())

        last_hs = self.fusion(torch.cat(hs))[0]

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output, hs_detach


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=5, layers=2,
                 relu_dropout=0.1, embed_dropout=0.3,
                 attn_dropout=0.25, res_dropout=0.1):
        super(Classifier, self).__init__()
        self.bone = TransformerEncoder(embed_dim=in_dim, num_heads=num_heads,
                                       layers=layers, attn_dropout=attn_dropout, res_dropout=res_dropout,
                                       relu_dropout=relu_dropout, embed_dropout=embed_dropout)

        self.proj1 = nn.Linear(in_dim, in_dim)
        self.out_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.bone(x)
        x = self.proj1(x[0])
        x = F.relu(self.proj1(x))
        x = self.out_layer(x)
        return x


class ClassifierGuided(nn.Module):
    def __init__(self, output_dim, num_mod, proj_dim=30, num_heads=5, layers=5,
                 relu_dropout=0.1, embed_dropout=0.3, res_dropout=0.1, attn_dropout=0.25):
        super(ClassifierGuided, self).__init__()
        # Classifiers
        self.num_mod = num_mod
        self.classifers = nn.ModuleList([
            Classifier(in_dim=proj_dim, out_dim=output_dim, layers=layers,
                       num_heads=num_heads, attn_dropout=attn_dropout, res_dropout=res_dropout,
                       relu_dropout=relu_dropout, embed_dropout=embed_dropout)
            for _ in range(self.num_mod)
        ])

    def cal_coeff(self, dataset, y, cls_res):
        acc_list = list()

        if dataset in ['mosi', 'mosei']:
            for r in cls_res:
                acc = train_eval_senti(r, y)
                acc_list.append(acc)
        elif dataset == 'iemo':
            for r in cls_res:
                acc = train_eval_iemo(r, y)
                acc_list.append(acc)
        elif dataset == 'food':
            for r in cls_res:
                acc = train_eval_food(r, y)
                acc_list.append(acc)

        return acc_list

    def forward(self, x):
        self.cls_res = list()
        for i in range(len(x)):
            self.cls_res.append(self.classifers[i](x[i]))
        return self.cls_res
