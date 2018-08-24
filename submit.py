import os
from PIL import Image
import cv2
import torch
import numpy as np
from skimage.transform import resize

from .predict import predict_img
from .utils import rle_encode
from .unet import UNet

# source: https://github.com/EdwardTyantov/ultrasound-nerve-segmentation/blob/master/submission.py
def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return res#' '.join([str(r) for r in res])

def prep(img):
    img = img.astype('float32')
    img = cv2.resize(img, (580, 420), interpolation=cv2.INTER_NEAREST) 
    img = cv2.threshold(img, 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    return img


def submit(net, gpu=False, results_dir='./', save=False):
    """Used for Kaggle submission: predicts and encode all test images"""
    dir = 'data/test/'

    N = len(list(os.listdir(dir)))
    filename = results_dir + 'SUBMISSION.csv'
    pred_dir = results_dir+'pred/'
    if os.path.exists(filename):
        os.remove(filename)
    if save and not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    with open(filename, 'a') as f:
        f.write('img,pixels\n')
        for index, i in enumerate(os.listdir(dir)):
            print('{}/{}'.format(index, N))

            img = Image.open(dir + i)

            mask = predict_img(net, img, use_gpu=gpu)
            mask = prep(mask)
            if save:
                cv2.imwrite(pred_dir+i, mask*255)
            #enc = rle_encode(mask)
            enc = run_length_enc(mask)
            f.write('{},{}\n'.format(i.split(".")[0], ' '.join(map(str, enc))))


if __name__ == '__main__':
    net = UNet(3, 1).cuda()
    net.load_state_dict(torch.load('results/run1_pytorch/CP50.pth'))
    submit(net, True)
