import os
import torch
import argparse
from torch.backends import cudnn
# from models.SFNet import build_net
from models.PiM_multioutput import UMamba_full
from eval_multioutput import _eval
from train_multioutput import _train


def main(args):
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    model = UMamba_full()
    # print(model)

    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Directories
    parser.add_argument('--model_name', default='TDNet', type=str)
    parser.add_argument('--dataroot', type=str, default='')
    parser.add_argument('--task_name', type=str, default='raincityscape')

    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--resume', type=str, default='')

    # Test
    parser.add_argument('--test_model', type=str, default='')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])
    parser.add_argument('--win_size', default=0, type=int, help='window size')
    parser.add_argument('--overlap_size', default=0, type=int, help='overlap size')

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, 'Training-Results/', args.task_name)
    # NOTE 2025年10月11日
    if args.mode=='train':
        args.result_dir = os.path.join('results/', args.model_name, 'images', args.task_name)
    elif args.mode=='test':
        args.result_dir = os.path.join('saved_images/', args.task_name)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    command = 'cp ' + 'models/layers.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'models/PiM_multioutput.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'train_multioutput.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'main_multioutput.py ' + args.model_save_dir
    os.system(command)
    print(args)
    main(args)
