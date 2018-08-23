import os
from PIL import Image
import cv2
import torch
import numpy as np

from .predict import predict_img
from .utils import rle_encode
from .unet import UNet


def submit(net, gpu=False, results_dir='./'):
    """Used for Kaggle submission: predicts and encode all test images"""
    dir = 'data/test/'

    N = len(list(os.listdir(dir)))
    filename = results_dir + 'SUBMISSION.csv'
    if os.path.exists(filename):
        os.remove('SUBMISSION.csv')
    with open(filename, 'a') as f:
        f.write('img,pixels\n')
        for index, i in enumerate(os.listdir(dir)):
            print('{}/{}'.format(index, N))

            img = Image.open(dir + i)

            mask = predict_img(net, img, use_gpu=gpu)
            mask = cv2.resize(np.array(mask*255.0, dtype=np.uint8), dsize=(580, 420), interpolation=cv2.INTER_NEAREST)
            enc = rle_encode(mask)
            f.write('{},{}\n'.format(i.split(".")[0], ' '.join(map(str, enc))))


if __name__ == '__main__':
    net = UNet(3, 1).cuda()
    net.load_state_dict(torch.load('results/run1_pytorch/CP50.pth'))
    submit(net, True)
