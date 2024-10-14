import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.checkpoint import checkpoint
import copy
import os

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MoTE(torch.nn.Module):

    def __init__(self, d_model: int, num_experts=4):
        super(MoTE, self).__init__()
        self.num_experts = num_experts

        # initialize experts
        fc_up = nn.Linear(d_model, d_model * 4)
        self.fc_up = torch.nn.ModuleList([copy.deepcopy(fc_up) for i in range(num_experts)])

        self.gelu = QuickGELU()

        fc_dn = nn.Linear(d_model * 4, d_model)
        self.fc_dn = torch.nn.ModuleList([copy.deepcopy(fc_dn) for i in range(num_experts)])

        self.mote_fusion_weight = torch.nn.Parameter(torch.zeros(self.num_experts), requires_grad=False)
        self.expert_index = 0

    def mote_experts_fusion(self, evaluation=False):
        # ============ specific weight ==============
        weight = torch.arange(start=1,end=self.num_experts+1, dtype=torch.float)
        weight_candi = [-9.6, -4.8, -2.4, -1.2, -0.6, 0.6, 1.2, 2.4, 4.8, 9.6, float('inf')]
        tau_idx = torch.randint(low=0, high=len(weight_candi), size=(1,)).item()
        tau = weight_candi[tau_idx]
        if not evaluation:
            weight = F.softmax(weight/tau, dim=-1)
        else:
            weight = F.softmax(self.mote_fusion_weight, dim=-1)

        self.fc_up_dict = {"weight": 0, "bias": 0}
        self.fc_dn_dict = {"weight": 0, "bias": 0}

        # weights merging
        for idx in range(self.num_experts):
            fc_up_single = self.fc_up[idx]
            fc_dn_single = self.fc_dn[idx]

            for s_name, s_param in fc_up_single.named_parameters():
                if "weight" in s_name:
                    p_name = "weight"
                    self.fc_up_dict[p_name] = self.fc_up_dict[p_name] + (weight[idx] * s_param)
                elif "bias" in s_name:
                    p_name = "bias"
                    self.fc_up_dict[p_name] = self.fc_up_dict[p_name] + (weight[idx] * s_param)
                else:
                    raise NotImplementedError

            for s_name, s_param in fc_dn_single.named_parameters():
                if "weight" in s_name:
                    p_name = "weight"
                    self.fc_dn_dict[p_name] = self.fc_dn_dict[p_name] + (weight[idx] * s_param)
                elif "bias" in s_name:
                    p_name = "bias"
                    self.fc_dn_dict[p_name] = self.fc_dn_dict[p_name] + (weight[idx] * s_param)
                else:
                    raise NotImplementedError
        
    def forward(self, x: torch.Tensor, regularization: bool = False):
        if self.fc_up[0].training and self.fc_dn[0].training:
            if regularization:
                self.mote_experts_fusion(evaluation=False)
                x = F.linear(x, self.fc_up_dict["weight"], self.fc_up_dict["bias"])
                x = self.gelu(x)
                x = F.linear(x, self.fc_dn_dict["weight"], self.fc_dn_dict["bias"])
            else:
                # ================ multinomial gate =================
                weights = torch.arange(start=1,end=self.num_experts+1, dtype=torch.float)
                weights = F.softmax(weights/1.0, dim=-1)
                expert_idx = torch.multinomial(input=weights, num_samples=1).item()
                
                self.expert_index = expert_idx
                # print('expert_idx: ', expert_idx)

                x = self.fc_up[expert_idx](x)
                x = self.gelu(x)
                x = self.fc_dn[expert_idx](x)
        else:
            self.mote_experts_fusion(evaluation=True)
            x = F.linear(x, self.fc_up_dict["weight"], self.fc_up_dict["bias"])
            x = self.gelu(x)
            x = F.linear(x, self.fc_dn_dict["weight"], self.fc_dn_dict["bias"])
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, num_experts: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)

        if num_experts >= 1:
            self.mlp = MoTE(d_model, num_experts=num_experts)
        else:
            self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))

        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, regularization: bool):
        x = x + self.attention(self.ln_1(x))
        if regularization:
            x = x + self.mlp(self.ln_2(x), regularization)
        else:
            x = x + self.mlp(self.ln_2(x))
        return x



class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, num_experts: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, num_experts, attn_mask) for _ in range(layers)])
        self.grad_checkpointing = False

    def forward(self, x: torch.Tensor, regularization: bool):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, regularization)
            else:
                x = r(x, regularization)
        return x


class video_header(nn.Module):
    def __init__(self, vid_head, interaction, clip_state_dict, temporal_layer=3, num_experts=4, spe_cls_feature=None):
        super().__init__()
        self.vid_header = vid_head
        self.interaction = interaction     
        self.mse_criterion = nn.MSELoss()
        if spe_cls_feature is None:
            # finetune on k400
            self.spe_cls_feature = nn.Parameter(torch.zeros(400,clip_state_dict["text_projection"].shape[1]), requires_grad=False)
        else:
            self.spe_cls_feature = nn.Parameter(spe_cls_feature, requires_grad=False)   
        
        assert vid_head in ["None", "Transf"]

        if self.vid_header == "Transf":
            embed_dim = clip_state_dict["text_projection"].shape[1]

            context_length = clip_state_dict["positional_embedding"].shape[0]
            vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64

            transformer_layers = len(
                set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)

            self.transformer = TemporalTransformer(width=embed_dim, layers=temporal_layer, heads=transformer_heads, num_experts=num_experts)
            print('=============== num temporal transformer layer: ',temporal_layer, '===============')

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def agg_video_feat(self, x, gen_cls_feat = None, regularization: bool = False):
        b, t, c = x.size()
        x = x.contiguous()
        if self.vid_header == "None":
            pass

        elif self.vid_header == "Transf":
            # Temporal Feature Modulation
            if gen_cls_feat is not None:
                num_esti = 5
                scale_param = 0.05

                # spatial features from clip part
                x_clip = x.mean(dim=1,keepdim=False) # [bs, dim]
                x_clip = x_clip / x_clip.norm(dim=-1,keepdim=True)
                bs,c_ = x_clip.size() 

                # estimate gen_feat and spe_feat
                gen_logits = x_clip @ gen_cls_feat.t() # [bs, gen_num_class]
                spe_logits = x_clip @ self.spe_cls_feature.t() # [bs, spe_num_class]
                
                gen_values, gen_indices = gen_logits.topk(num_esti,-1,True,True) # values, indices: [bs, num_esti]
                gen_feat_esti = torch.gather(input=gen_cls_feat.unsqueeze(0).expand(bs,-1,-1), dim=1, \
                                        index = gen_indices.unsqueeze(-1).expand(-1,-1,c_)) # [bs, num_esti, dim]
                spe_values, spe_indices = spe_logits.topk(num_esti,-1,True,True) # values, indices: [bs, num_esti]
                spe_feat_esti = torch.gather(input=self.spe_cls_feature.unsqueeze(0).expand(bs,-1,-1), dim=1, \
                                        index = spe_indices.unsqueeze(-1).expand(-1,-1,c_)) # [bs, num_esti, dim]

                gen_spe_sim = torch.bmm(gen_feat_esti,spe_feat_esti.permute(0,2,1)) # [bs,num_esti, num_esti]
                gen_spe_sim, indi = torch.max(gen_spe_sim,dim=-1)
                gen_spe_sim=1/torch.exp((1-gen_spe_sim.mean(dim=-1))/scale_param)
                gen_spe_sim=gen_spe_sim.unsqueeze(-1).unsqueeze(-1)
            else:
                gen_spe_sim=torch.ones(1,device=x.device)

            x_original = x
            seq_length = t
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            x = x + frame_position_embeddings

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x, regularization)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = gen_spe_sim * x.type(x_original.dtype) + x_original
        else:
            raise ValueError('Unknown temporal modeling header: {}'.format(self.vid_header))
        return x


    def get_logits(self, vid_emb, cls_emb):
        if self.interaction == 'DP':
            vid_emb = vid_emb.mean(dim=1, keepdim=False)
            vid_emb = vid_emb / vid_emb.norm(dim=-1, keepdim=True)
            cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
            logit = vid_emb @ cls_emb.t()  
        else:
            raise NotImplementedError
        return logit

    def forward(self, vid_emb, cls_emb):
        if self.training:
            vid_emb_expert = self.agg_video_feat(vid_emb, regularization=False)
            logits = self.get_logits(vid_emb_expert, cls_emb)

            vid_emb_reg = self.agg_video_feat(vid_emb, regularization=True)
            logits_reg = self.get_logits(vid_emb_reg, cls_emb)

            mse_loss = self.mse_criterion(vid_emb_reg, vid_emb)
            return logits, logits_reg, mse_loss
        else:
            vid_emb = self.agg_video_feat(vid_emb, cls_emb, regularization=False)
            logits = self.get_logits(vid_emb, cls_emb)
            return logits

class VideoCLIP(nn.Module):
    def __init__(self, clip_model, n_seg) :
        super(VideoCLIP, self).__init__()
        self.visual = clip_model.visual
        self.n_seg = n_seg
        self.logit_scale = clip_model.logit_scale

    def forward(self, image):
        # CLIP encode images
        image_emb = self.encode_image(image) # [BS, T, C]
        return image_emb, self.logit_scale.exp()

    def encode_image(self, image):
        bt = image.size(0) # [BS*T, C, H, W]
        b = bt // self.n_seg
        image_emb = self.visual(image) # [BS*T, C]
        image_emb = image_emb.view(b, self.n_seg, -1) # [BS, T, C]
        return image_emb

