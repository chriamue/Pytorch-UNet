import os
from PIL import Image

import torch

from .predict import predict_img
from .utils import rle_encode
from .unet import UNet


def submit(net, gpu=False):
    """Used for Kaggle submission: predicts and encode all test images"""
    dir = 'data/test/'

    N = len(list(os.listdir(dir)))
    with open('SUBMISSION.csv', 'a') as f:
        f.write('img,pixels\n')
        for index, i in enumerate(os.listdir(dir)):
            print('{}/{}'.format(index, N))

            img = Image.open(dir + i)

            mask = predict_img(net, img, use_gpu=gpu)
            enc = rle_encode(mask)
            f.write('{},{}\n'.format(i.split(".")[0], ' '.join(map(str, enc))))


if __name__ == '__main__':
    net = UNet(3, 1).cuda()
    net.load_state_dict(torch.load('results/run1_pytorch/CP50.pth'))
    submit(net, True)
