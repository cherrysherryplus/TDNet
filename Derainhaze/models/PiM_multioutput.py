# Based on SFNet, Restormer, TransMamba, C2PNet, NeRD-Rain

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Definition of IND (Implicit Neural Deraining module)
from networks.ind import INR
# Definition of PAD (Prior-adaptive Dehazing module)
from networks.pnd import PDU
# Definition of SRM (Scene Restoration Module)
from networks.srmamba import SRMamba

from networks.layernorm import LayerNorm
from models.layers import BasicConv



# Definition of SRB (Scene Restoration Block)
class MambaBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(MambaBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.mamba1 = SRMamba(dim, reverse=False)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.mamba2 = SRMamba(dim, reverse=True)
        
    def forward(self, x):
        x = x + self.mamba1(self.norm1(x))
        x = x + self.mamba2(self.norm2(x))
        return x
    

# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


# Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
    def forward(self, x):
        return self.body(x)


# Definition of TDNet
class UMamba_full(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 28,
        num_blocks = [1,2,4,8], 
        num_refinement_blocks = 1,
        heads = [4,4,8,8],
        ffn_expansion_factor = 1.6667,
        bias = False,
        LayerNorm_type = 'WithBias',
    ):

        super(UMamba_full, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.inr1_patch_embed = OverlapPatchEmbed(inp_channels, dim * 2)
        self.inr2_patch_embed = OverlapPatchEmbed(inp_channels, dim * 4)
        self.pdu1 = PDU(dim * 2)
        self.pdu2 = PDU(dim * 4)

        self.encoder_level1 = nn.Sequential(*[MambaBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.INR1 = INR(dim=3).cuda()
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[MambaBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.INR2 = INR(dim=3).cuda()
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[MambaBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[MambaBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[MambaBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[MambaBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[MambaBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[MambaBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.ConvsOut = nn.ModuleList([
            BasicConv(dim*4, out_channels, kernel_size=3, stride=1, relu=False, bias=True),  # for level3, 64x64
            BasicConv(dim*2, out_channels, kernel_size=3, stride=1, relu=False, bias=True),  # for level2, 128x128
        ])


    def forward(self, inp_img):
        inp_img_mid = F.interpolate(inp_img, scale_factor=0.5)
        inp_img_small = F.interpolate(inp_img, scale_factor=0.25)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        inr1 = self.INR1(inp_img_mid)
        # NOTE 0813 add inr output
        # inr1_out = inr1 + inp_img_mid
        inr1_cp = inr1.clone()
        
        inr1 = self.inr1_patch_embed(inr1)
        inr1 = self.pdu1(inr1)
        # inp_enc_level2 = self.FAM2(inp_enc_level2, inr1)
        inp_enc_level2 = inp_enc_level2 + inr1
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        out_enc_level2 = out_enc_level2

        inp_enc_level3 = self.down2_3(out_enc_level2)
        inr2 = self.INR2(inp_img_small)
        # NOTE 0813 add inr output
        # inr2_out = inr2 + inp_img_small
        inr2_cp = inr2.clone()

        inr2 = self.inr2_patch_embed(inr2)
        inr2 = self.pdu2(inr2)
        inp_enc_level3 = inp_enc_level3 + inr2
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
    
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
    
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        # out_level3 = self.ConvsOut[0](out_dec_level3) + inp_img_small # 直接+inr2?
        out_level3 = self.ConvsOut[0](out_dec_level3) + inr2_cp
        outputs = [out_level3]

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 
    
        # out_level2 = self.ConvsOut[1](out_dec_level2) + inp_img_mid
        out_level2 = self.ConvsOut[1](out_dec_level2) + inr1_cp
        outputs.append(out_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
    
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        
        outputs.append(out_dec_level1)
        
        return outputs
    

# Definition of TDNet with only the final output (for testing FPS)
class UMamba_full_fps(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 28,
        num_blocks = [1,2,4,8], 
        num_refinement_blocks = 1,
        heads = [4,4,8,8],
        ffn_expansion_factor = 1.6667,
        bias = False,
        LayerNorm_type = 'WithBias',
    ):

        super(UMamba_full_fps, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.inr1_patch_embed = OverlapPatchEmbed(inp_channels, dim * 2)
        self.inr2_patch_embed = OverlapPatchEmbed(inp_channels, dim * 4)
        self.pdu1 = PDU(dim * 2)
        self.pdu2 = PDU(dim * 4)

        self.encoder_level1 = nn.Sequential(*[MambaBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.INR1 = INR(dim=3).cuda()
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[MambaBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.INR2 = INR(dim=3).cuda()
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[MambaBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[MambaBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[MambaBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[MambaBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[MambaBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[MambaBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


    def forward(self, inp_img):
        inp_img_mid = F.interpolate(inp_img, scale_factor=0.5)
        inp_img_small = F.interpolate(inp_img, scale_factor=0.25)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        inr1 = self.INR1(inp_img_mid)
        # NOTE 0813 add inr output
        # inr1_out = inr1 + inp_img_mid
        inr1_cp = inr1.clone()
        
        inr1 = self.inr1_patch_embed(inr1)
        inr1 = self.pdu1(inr1)
        inp_enc_level2 = inp_enc_level2 + inr1
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        out_enc_level2 = out_enc_level2

        inp_enc_level3 = self.down2_3(out_enc_level2)
        inr2 = self.INR2(inp_img_small)
        # NOTE 0813 add inr output
        # inr2_out = inr2 + inp_img_small
        inr2_cp = inr2.clone()

        inr2 = self.inr2_patch_embed(inr2)
        inr2 = self.pdu2(inr2)
        # inp_enc_level3 = self.FAM1(inp_enc_level3, inr2)
        inp_enc_level3 = inp_enc_level3 + inr2
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
    
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
    
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
    
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        
        return out_dec_level1



if __name__ == '__main__':
    device_id0 = 'cuda:0'
    
    x = torch.randn(1, 3, 256, 256).cuda(device_id0)
    B, C, H, W = x.shape
    net = UMamba_full().cuda(device_id0)
    out = net(x)
    print("Output shapes:", [o.shape for o in out])

    try:
        from calflops import calculate_flops

        model = UMamba_full().cuda(device_id0)
        batch_size = 1
        input_shape = (batch_size, 3, 256, 256)
        flops, macs, params = calculate_flops(model=model,
                                            input_shape=input_shape,
                                            output_as_string=True,
                                            output_precision=4, 
                                            print_results=False)
        print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
    except Exception as e:
        print(e)
