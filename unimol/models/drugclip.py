# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.data import Dictionary
from unicore.models import (BaseUnicoreModel, register_model,
                            register_model_architecture)
from unicore.modules import LayerNorm
import unicore


from .transformer_encoder_with_pair import TransformerEncoderWithPair
from .unimol import NonLinearHead, UniMolModel, base_architecture

logger = logging.getLogger(__name__)


@register_model("drugclip")
class BindingAffinityModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--mol-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pocket-encoder-layers",
            type=int,
            help="pocket encoder layers",
        )
        parser.add_argument(
            "--recycling",
            type=int,
            default=1,
            help="recycling nums of decoder",
        )


    def __init__(self, args, mol_dictionary, pocket_dictionary):
        super().__init__()
        drugclip_architecture(args)
        self.args = args
        self.mol_model = UniMolModel(args.mol, mol_dictionary)
        self.pocket_model = UniMolModel(args.pocket, pocket_dictionary)

        self.cross_distance_project = NonLinearHead(
            args.mol.encoder_embed_dim * 2 + args.mol.encoder_attention_heads, 1, "relu"
        )
        self.holo_distance_project = DistanceHead(
            args.mol.encoder_embed_dim + args.mol.encoder_attention_heads, "relu"
        )
        
        self.mol_project = NonLinearHead(
            args.mol.encoder_embed_dim, 128, "relu"
        )

        
        lshape = [1]
        if "lip" in args.loss:
            self.logit_scale = nn.Parameter(torch.ones(lshape) * args.init_logit_scale)
            self.logit_bias = nn.Parameter(torch.ones(lshape) * args.init_logit_bias) if args.init_logit_bias is not None else None
        else:
            self.logit_scale = nn.Parameter(torch.ones([1], device="cuda") * np.log(14))
            
        
        self.pocket_project = NonLinearHead(
            args.pocket.encoder_embed_dim, 128, "relu"
        )
        
        self.fuse_project = NonLinearHead(
            256, 1, "relu"
        )
        self.classification_head = nn.Sequential(
            nn.Linear(args.pocket.encoder_embed_dim + args.pocket.encoder_embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary, task.pocket_dictionary)


    def get_dist_features(self, dist, et, flag):
        if flag == "mol":
            n_node = dist.size(-1)
            gbf_feature = self.mol_model.gbf(dist, et)
            gbf_result = self.mol_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        else:
            n_node = dist.size(-1)
            gbf_feature = self.pocket_model.gbf(dist, et)
            gbf_result = self.pocket_model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates

    def encode(self, src_tokens, src_distance, src_edge_type, component="pocket", return_logit_param=False, return_cls_rep=False, **kwargs):
        model = self.mol_model if component == "mol" else self.pocket_model
        project = self.mol_project if component == "mol" else self.pocket_project
        padding_mask = src_tokens.eq(model.padding_idx)
        x = model.embed_tokens(src_tokens)
        graph_attn_bias = self.get_dist_features(
            src_distance, src_edge_type, component
        )
        mol_outputs = model.encoder(
            x, padding_mask=padding_mask, attn_mask=graph_attn_bias
        )
        encoder_rep = mol_outputs[0]
        encoder_pair_rep = mol_outputs[1]

        rep =  encoder_rep[:,0,:]

        emb = project(rep)
        emb = emb / emb.norm(dim=1, keepdim=True)
        if return_cls_rep:
            return emb, rep
        return emb


    def forward(self, inputs, components, **kwargs):
        embs = []
        for i, component in enumerate(components):
            emb = self.encode(*inputs[i], component=component)
            embs.append(emb)
            logits_params = {"logit_scale": self.logit_scale.exp()}
            if hasattr(self, "logit_bias") and self.logit_bias is not None:
                logits_params["logit_bias"] = self.logit_bias * 1.0
        return embs, logits_params




class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        x[x == float("-inf")] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x




@register_model_architecture("drugclip", "drugclip")
def drugclip_architecture(args):

    parser = argparse.ArgumentParser()
    args.mol = parser.parse_args([])
    args.pocket = parser.parse_args([])

    args.mol.encoder_layers = getattr(args, "mol_encoder_layers", 15)
    args.mol.encoder_embed_dim = getattr(args, "mol_encoder_embed_dim", 512)
    args.mol.encoder_ffn_embed_dim = getattr(args, "mol_encoder_ffn_embed_dim", 2048)
    args.mol.encoder_attention_heads = getattr(args, "mol_encoder_attention_heads", 64)
    args.mol.dropout = getattr(args, "mol_dropout", 0.1)
    args.mol.emb_dropout = getattr(args, "mol_emb_dropout", 0.1)
    args.mol.attention_dropout = getattr(args, "mol_attention_dropout", 0.1)
    args.mol.activation_dropout = getattr(args, "mol_activation_dropout", 0.0)
    args.mol.pooler_dropout = getattr(args, "mol_pooler_dropout", 0.0)
    args.mol.max_seq_len = getattr(args, "mol_max_seq_len", 512)
    args.mol.activation_fn = getattr(args, "mol_activation_fn", "gelu")
    args.mol.pooler_activation_fn = getattr(args, "mol_pooler_activation_fn", "tanh")
    args.mol.post_ln = getattr(args, "mol_post_ln", False)
    args.mol.masked_token_loss = -1.0
    args.mol.masked_coord_loss = -1.0
    args.mol.masked_dist_loss = -1.0
    args.mol.x_norm_loss = -1.0
    args.mol.delta_pair_repr_norm_loss = -1.0

    args.pocket.encoder_layers = getattr(args, "pocket_encoder_layers", 15)
    args.pocket.encoder_embed_dim = getattr(args, "pocket_encoder_embed_dim", 512)
    args.pocket.encoder_ffn_embed_dim = getattr(
        args, "pocket_encoder_ffn_embed_dim", 2048
    )
    args.pocket.encoder_attention_heads = getattr(
        args, "pocket_encoder_attention_heads", 64
    )
    args.pocket.dropout = getattr(args, "pocket_dropout", 0.1)
    args.pocket.emb_dropout = getattr(args, "pocket_emb_dropout", 0.1)
    args.pocket.attention_dropout = getattr(args, "pocket_attention_dropout", 0.1)
    args.pocket.activation_dropout = getattr(args, "pocket_activation_dropout", 0.0)
    args.pocket.pooler_dropout = getattr(args, "pocket_pooler_dropout", 0.0)
    args.pocket.max_seq_len = getattr(args, "pocket_max_seq_len", 512)
    args.pocket.activation_fn = getattr(args, "pocket_activation_fn", "gelu")
    args.pocket.pooler_activation_fn = getattr(
        args, "pocket_pooler_activation_fn", "tanh"
    )
    args.pocket.post_ln = getattr(args, "pocket_post_ln", False)
    args.pocket.masked_token_loss = -1.0
    args.pocket.masked_coord_loss = -1.0
    args.pocket.masked_dist_loss = -1.0
    args.pocket.x_norm_loss = -1.0
    args.pocket.delta_pair_repr_norm_loss = -1.0

    base_architecture(args)


class ArgMaxAdaptor(nn.Module):
    def __init__(self, *args, add_mol_linear=False, **kwargs):
        super().__init__(*args, **kwargs)
        if add_mol_linear:
            self.mol_linear = nn.Linear(128, 128)

    def forward(self, mol_embs=None, pocket_embs=None, **kwargs):
        key_padding_mask = pocket_embs.eq(0).all(dim=-1)
        res = torch.matmul(pocket_embs, mol_embs.transpose(-2, -1))
        res = res.masked_fill(key_padding_mask.unsqueeze(-1), -1e4)
        pocket_idx = torch.argmax(res, dim=1)
        agg_embs = pocket_embs[torch.arange(mol_embs.shape[0]), pocket_idx[torch.eye(mol_embs.shape[0], dtype=torch.bool)]]
        if hasattr(self, "mol_linear"):
            mol_embs = self.mol_linear(mol_embs)
            return agg_embs, mol_embs
        return agg_embs
    
    def infer_one(self, mol_rep=None, pocket_rep=None, **kwargs):
        if hasattr(self, "mol_linear"):
            mol_embs = self.mol_linear(mol_rep)
            return pocket_rep, mol_embs
        return pocket_rep


class CrossAttAdapter(nn.Module):
    def __init__(self, *args, agg_project=False, add_mol_linear=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptor = nn.MultiheadAttention(512, 8, batch_first=True)
        if agg_project:
            self.agg_project = NonLinearHead(512, 128, "relu")
        if add_mol_linear:
            # self.mol_linear = NonLinearHead(128, 128, "relu", 32)
            self.mol_linear = nn.Linear(128, 128)
            
    def forward(self, mol_embs=None, pocket_embs=None, pocket_rep=None, return_attn=False, **kwargs):
        if pocket_rep is not None:
            pocket_padding_mask = pocket_rep.eq(0).all(dim=-1)
        else:
            pocket_padding_mask = pocket_embs.eq(0).all(dim=-1)
        agg_embs = []
        attn_l = []
        for i, mf in enumerate(mol_embs):
            embs, attn = self.adaptor(mf.repeat(pocket_embs.shape[0], 1, 1), pocket_embs, pocket_embs, key_padding_mask=pocket_padding_mask)
            agg_embs.append(embs)
            attn_l.append(attn)
        agg_embs = torch.cat(agg_embs, dim=-2)
        attn_l = torch.cat(attn_l, dim=-2)
        if hasattr(self, "agg_project"):
            agg_embs = self.agg_project(agg_embs)
        if return_attn:
            agg_embs = (agg_embs, attn_l)
        if hasattr(self, "mol_linear"):
            mol_embs = self.mol_linear(mol_embs)
            return agg_embs, mol_embs
        return agg_embs
    
    def infer_one(self, mol_rep=None, pocket_rep=None, **kwargs):
        if len(mol_rep.shape) == 2:
            mol_rep = mol_rep.unsqueeze(0)
        if len(pocket_rep.shape) == 2:
            pocket_rep = pocket_rep.unsqueeze(0)
        agg_embs, agg_weights = self.adaptor(mol_rep, pocket_rep, pocket_rep)
        if hasattr(self, "agg_project"):
            agg_embs = self.agg_project(agg_embs)
        if hasattr(self, "mol_linear"):
            mol_embs = self.mol_linear(mol_rep)
            return (agg_embs, mol_embs), agg_weights
        return agg_embs, agg_weights


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0., learn_temperature=False):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones([1]) * temperature)
        self.temperature.requires_grad = learn_temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q * self.temperature, k.transpose(2, 3))
        attn_ = attn

        if mask is not None:
            if len(mask.shape) < len(attn.shape):
                mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask, -1e4)  # for fp16 support not using -1e9
        attn = self.dropout(F.softmax(attn, dim=-1))
        if (attn.sum(-1) < 0.1).any():
            print("attn is all zero")
            print("mask", mask[torch.where(attn.sum(-1) < 0.1)])
            print("attn", attn_[torch.where(attn.sum(-1) < 0.1)])
            print(attn_)
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, bias=True, temperature=None, learn_temperature=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=bias)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=bias)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=bias)
        self.attention = ScaledDotProductAttention(temperature=1 / d_k ** 0.5 if temperature is None else temperature, attn_dropout=0., learn_temperature=learn_temperature)

        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, key_padding_mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # residual = q
        # q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1)   # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=key_padding_mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        # q += residual
        if torch.isnan(q).any():
            print("out has nan")
        if torch.isnan(attn).any():
            print("attn has nan")
        return q, attn


class IdenticalCrossAttAdapter(CrossAttAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.adaptor = MultiheadAttention(512, 1, batch_first=True, kdim=512, vdim=512, bias=False)
        self.adaptor = MultiHeadAttention(1, 128, 128, 128, temperature=5)
        self._initialize_attention_weights()

    def _initialize_attention_weights(self):
        with torch.no_grad():
            nn.init.eye_(self.adaptor.w_qs.weight)
            nn.init.eye_(self.adaptor.w_ks.weight)
            nn.init.eye_(self.adaptor.w_vs.weight)
            nn.init.eye_(self.adaptor.fc.weight)
            nn.init.zeros_(self.adaptor.w_qs.bias)
            nn.init.zeros_(self.adaptor.w_ks.bias)
            nn.init.zeros_(self.adaptor.w_vs.bias)
            nn.init.zeros_(self.adaptor.fc.bias)
            if hasattr(self.adaptor, "mol_linear"):
                nn.init.eye_(self.adaptor.mol_linear.weight)
                nn.init.zeros_(self.adaptor.mol_linear.bias)


@register_model("drugclip_adaptor")
class DrugClipAdaptor(BindingAffinityModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--adaptor-type", type=str, default="max", choices=["cross_attention", "identical_cross_attention", "max", "self_attention", "res_max_self_attention"])
        parser.add_argument("--add-mol-linear", action="store_true", default=False, help="add mol linear layer")

    def __init__(self, args, mol_dictionary, pocket_dictionary):
        super().__init__(args, mol_dictionary, pocket_dictionary)
        if args.adaptor_type == "cross_attention":
            self.adaptor = CrossAttAdapter()
        elif args.adaptor_type == "identical_cross_attention":
            self.adaptor = IdenticalCrossAttAdapter(add_mol_linear=args.add_mol_linear)
        elif args.adaptor_type == "max":
            self.adaptor = ArgMaxAdaptor(add_mol_linear=args.add_mol_linear)
        
    def forward(
        self,
        inputs=None,
        components=None,
        mol_inputs=None,
        pocket_embs=None,
        pocket_rep=None,
        mol_embs=None,
        return_attn=False,
    ):
        if mol_inputs:
            mol_embs, mol_rep = self.encode(*mol_inputs, component="mol", return_cls_rep=True)
        else:
            mol_rep = None
        if pocket_embs is None:
            pocket_embs = self.pocket_project(pocket_rep)
            pocket_embs = pocket_embs / pocket_embs.norm(dim=-1, keepdim=True)
        agg_embs = self.adaptor(mol_embs=mol_embs, pocket_embs=pocket_embs, mol_rep=mol_rep, pocket_rep=pocket_rep, return_attn=return_attn)
        if hasattr(self.adaptor, "mol_linear"):
            agg_embs, mol_embs = agg_embs
            mol_embs = mol_embs / mol_embs.norm(dim=-1, keepdim=True)
        if return_attn:
            agg_embs, attn = agg_embs
        # if not hasattr(self, "agg_project"):
        #     agg_embs = self.pocket_project(agg_embs)
        agg_embs = agg_embs / agg_embs.norm(dim=-1, keepdim=True)
        if torch.isnan(agg_embs).any():
            print(f"agg_embs has nan at {torch.where(torch.isnan(agg_embs))}")
            raise
        if return_attn:
            agg_embs = (agg_embs, attn)
        return mol_embs, agg_embs, {"logit_scale": self.logit_scale.exp(), "logit_bias": self.logit_bias * 1.0}
    
    
@register_model_architecture("drugclip_adaptor", "drugclip_adaptor")
def drugclip_architecture(args):

    parser = argparse.ArgumentParser()
    args.mol = parser.parse_args([])
    args.pocket = parser.parse_args([])

    args.mol.encoder_layers = getattr(args, "mol_encoder_layers", 15)
    args.mol.encoder_embed_dim = getattr(args, "mol_encoder_embed_dim", 512)
    args.mol.encoder_ffn_embed_dim = getattr(args, "mol_encoder_ffn_embed_dim", 2048)
    args.mol.encoder_attention_heads = getattr(args, "mol_encoder_attention_heads", 64)
    args.mol.dropout = getattr(args, "mol_dropout", 0.1)
    args.mol.emb_dropout = getattr(args, "mol_emb_dropout", 0.1)
    args.mol.attention_dropout = getattr(args, "mol_attention_dropout", 0.1)
    args.mol.activation_dropout = getattr(args, "mol_activation_dropout", 0.0)
    args.mol.pooler_dropout = getattr(args, "mol_pooler_dropout", 0.0)
    args.mol.max_seq_len = getattr(args, "mol_max_seq_len", 512)
    args.mol.activation_fn = getattr(args, "mol_activation_fn", "gelu")
    args.mol.pooler_activation_fn = getattr(args, "mol_pooler_activation_fn", "tanh")
    args.mol.post_ln = getattr(args, "mol_post_ln", False)
    args.mol.masked_token_loss = -1.0
    args.mol.masked_coord_loss = -1.0
    args.mol.masked_dist_loss = -1.0
    args.mol.x_norm_loss = -1.0
    args.mol.delta_pair_repr_norm_loss = -1.0

    args.pocket.encoder_layers = getattr(args, "pocket_encoder_layers", 15)
    args.pocket.encoder_embed_dim = getattr(args, "pocket_encoder_embed_dim", 512)
    args.pocket.encoder_ffn_embed_dim = getattr(
        args, "pocket_encoder_ffn_embed_dim", 2048
    )
    args.pocket.encoder_attention_heads = getattr(
        args, "pocket_encoder_attention_heads", 64
    )
    args.pocket.dropout = getattr(args, "pocket_dropout", 0.1)
    args.pocket.emb_dropout = getattr(args, "pocket_emb_dropout", 0.1)
    args.pocket.attention_dropout = getattr(args, "pocket_attention_dropout", 0.1)
    args.pocket.activation_dropout = getattr(args, "pocket_activation_dropout", 0.0)
    args.pocket.pooler_dropout = getattr(args, "pocket_pooler_dropout", 0.0)
    args.pocket.max_seq_len = getattr(args, "pocket_max_seq_len", 512)
    args.pocket.activation_fn = getattr(args, "pocket_activation_fn", "gelu")
    args.pocket.pooler_activation_fn = getattr(
        args, "pocket_pooler_activation_fn", "tanh"
    )
    args.pocket.post_ln = getattr(args, "pocket_post_ln", False)
    args.pocket.masked_token_loss = -1.0
    args.pocket.masked_coord_loss = -1.0
    args.pocket.masked_dist_loss = -1.0
    args.pocket.x_norm_loss = -1.0
    args.pocket.delta_pair_repr_norm_loss = -1.0

    base_architecture(args)
