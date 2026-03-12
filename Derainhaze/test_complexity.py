import torch

import os

# from models.SFNet import build_net
from models.PiM_multioutput import UMamba_full

if __name__ == '__main__':
    mode = ['test', 'rainhaze_synscapes']
    # model_restoration = build_net(mode).cuda()
    model_restoration = UMamba_full().cuda()
    model_restoration.eval()
    device_id0 = 'cuda:0'
    # device_id0 = 'cpu'

    try:
        from calflops import calculate_flops
        with torch.no_grad():
            batch_size = 1
            input_shape = (batch_size, 3, 256, 256)
            flops, macs, params = calculate_flops(model=model_restoration,
                                                input_shape=input_shape,
                                                output_as_string=True,
                                                output_precision=4)
            print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
    except Exception as e:
        print(e)