import torch
import torch.nn as nn
import torch.nn.functional as F


def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list=[256, 256, 256]):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class INR(nn.Module):
    def __init__(self, dim, L=10, hidden_list=[256,256,256], local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.L = L                                  # 10；这一行没有用
        self.local_ensemble = local_ensemble        # True
        self.feat_unfold = feat_unfold              # True
        self.cell_decode = cell_decode              # True
        imnet_in_dim = dim                          # 3
        
        if self.feat_unfold:
            imnet_in_dim = dim * 10                 # 30  
        
        self.coord_enc = nn.Linear(2, 4 * L)        # 2,->40,
        imnet_in_dim += 4 * L                       # 70
        
        imnet_in_dim += 2                           # 72
        
        if self.cell_decode:
            imnet_in_dim += 2                       # 74

        self.imnet = MLP(imnet_in_dim, 3, hidden_list)

    def query_rgb(self, inp, coord, cell=None):
        feat = inp
        
        if self.feat_unfold:
            unfolded_feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            feat = torch.cat([feat, unfolded_feat], dim=1)  # [B, C*10, H, W]

        if self.local_ensemble:
            vx_lst = [-1, 0, 1]
            vy_lst = [-1, 0, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                
                bs, q = feat.shape[0], coord_.shape[1]
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                points_enc = self.coord_enc(q_coord) 
                
                rel_coord = coord_ - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                
                inp_concat = torch.cat([q_feat, points_enc, rel_coord], dim=-1)
                
                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp_concat = torch.cat([inp_concat, rel_cell], dim=-1)
                
                pred = self.imnet(inp_concat)
                preds.append(pred)
                
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        weights = torch.stack(areas, dim=0)  # [9, B, N]
        weights = F.softmax(-weights, dim=0)
        
        ret = 0
        for pred, weight in zip(preds, weights):
            ret = ret + pred * weight.unsqueeze(-1)
        
        B = inp.shape[0]
        H, W = inp.shape[2], inp.shape[3]
        return ret.permute(0, 2, 1).view(B, 3, H, W)

    def forward(self, inp):
        h, w = inp.shape[2], inp.shape[3]
        coord = make_coord((h, w)).cuda()
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        return self.query_rgb(inp, 
                            coord.unsqueeze(0).repeat(inp.shape[0], 1, 1),
                            cell.unsqueeze(0).repeat(inp.shape[0], 1, 1))
    
