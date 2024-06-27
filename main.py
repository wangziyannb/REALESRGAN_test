# #!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 6/24/2024 4:11 PM
# @Author  : Wang Ziyan
# @Email   : 1269586767@qq.com
# @File    : main.py
# @Software: PyCharm
import argparse
import os

import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
import numpy as np


def load_realesrgan(args, device):
    # init the model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    #
    # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    # netscale = 2
    # file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    # prepare wrapper class
    model_path = os.path.join('weights', 'RealESRGAN_x4plus.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
    dni_weight = None
    # wrapper class will load the weight based on the given model_path
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=False,
        device=device)
    # feeds model the input
    y = upsampler.model(torch.from_numpy(np.random.rand(5, 3, 255, 255)).float().to(device))
    print(y)



    # if args.face_enhance:  # Use GFPGAN for face enhancement
    from gfpgan import GFPGANer
    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=args.outscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler,
        device=device)
    return face_enhancer
    # return upsampler
    # return face_enhancer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument('-s', '--outscale', type=float, default=3.5, help='The final upsampling scale of the image')
    # parser.add_argument('--sr_path', type=str, default="./checkpoints/RealESRGAN_x4plus.pth",
    #                     help='Name of saved checkpoint to load super resolution weights from', required=False)
    parser.add_argument('--sr_path', type=str, default="./checkpoints/RealESRGAN_x4plus.pth",
                        help='Name of saved checkpoint to load super resolution weights from', required=False)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sr_model = load_realesrgan(args, device)
    img = np.random.rand(255, 255, 3)
    a, b, output = sr_model.enhance(img, has_aligned=False, only_center_face=False,
                                    paste_back=True)
    print(a, b, output)
