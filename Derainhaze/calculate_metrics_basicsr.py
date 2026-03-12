import cv2
import numpy as np
from tqdm import tqdm
import os
from os import path as osp
from skimage.color import rgb2ycbcr
from PIL import Image
import argparse


def to_y_channel(img):
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.

def reorder_image(img, input_order='HWC'):
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are '"'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    img = img.astype(np.float64)
    return img

def _convert_input_type_range(img):
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, 'f'but got {img_type}')
    return img

def _convert_output_type_range(img, dst_type):
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, 'f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)

def bgr2ycbcr(img, y_only=False):
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                            [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def calculate_psnr(img1,img2,crop_border,input_order='HWC',test_y_channel=False):
    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))

def _ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1,img2,crop_border,input_order='HWC',test_y_channel=False):
    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ''"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


#############################################################################
## Main
#############################################################################
parser = argparse.ArgumentParser(description='Calculate PSNR and SSIM for image datasets.')
# raincityscape raincityscape_pp rainhaze_synscapes
parser.add_argument('--dataset', type=str, default='raincityscape_pp', help='Dataset name')
parser.add_argument('--crop_border', type=int, default=8, help='Crop border size')
parser.add_argument('--suffix', type=str, default='', help='Suffix for GT image names')
parser.add_argument('--tile_strategy', type=str, default='', choices=['', 'slide-window', 'nerd-rain'])

args = parser.parse_args()

dataset = args.dataset
tile_strategy = args.tile_strategy
folder_gt = f'/root/workspace/d3npjrkp420c73baj4ig/datasets/{dataset}/test/clear'
if dataset == 'RW2AH':
    folder_gt = f'/root/workspace/d3npjrkp420c73baj4ig/datasets/{dataset}/test/target'
elif dataset == 'Rain200H' or dataset == 'HQNightRain-RS':
    folder_gt = f'/root/workspace/d3npjrkp420c73baj4ig/datasets/{dataset}/test/target'
folder_restored = f'saved_images/{dataset}' if not tile_strategy else f'saved_images/{dataset}_{tile_strategy}'


def main():
    psnr_all = []
    ssim_all = []
    img_list = sorted(scandir(folder_restored, recursive=False, full_path=True))

    for i, img_path in tqdm(enumerate(img_list), total=len(img_list)):
        img_restored = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_name, ext = osp.splitext(osp.basename(img_path))

        if 'raincityscape' in dataset:
            gt_path = osp.join(folder_gt, img_name.split('_rain_')[0] + args.suffix + ext)
        elif dataset == 'rainhaze_synscapes':
            gt_path = osp.join(folder_gt, img_name.split('_haze')[0] + args.suffix + ext)
        elif dataset == 'RW2AH' or dataset == 'Rain200H' or dataset == 'HQNightRain-RS':
            gt_path = osp.join(folder_gt, img_name + args.suffix + ext)
        else:
            gt_path = img_path

        print(gt_path)
        img_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        if img_gt.shape[-1] == 4:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGRA2BGR)
        if img_gt.ndim == 3 and img_gt.shape[2] == 3:
            img_gt = bgr2ycbcr(img_gt, y_only=True)
            img_restored = bgr2ycbcr(img_restored, y_only=True)
        psnr = calculate_psnr(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')
        ssim = calculate_ssim(img_gt * 255, img_restored * 255, crop_border=args.crop_border, input_order='HWC')

        print(psnr, ssim, os.path.basename(img_path))
        psnr_all.append(psnr)
        ssim_all.append(ssim)

    print(f"Average PSNR: {np.array(psnr_all).mean():.4f}")
    print(f"Average SSIM: {np.array(ssim_all).mean():.4f}")

if __name__ == "__main__":
    main()
    
    