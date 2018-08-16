#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os
import zipfile
from shutil import copyfile

import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def unzip(data_folder):
    """Unzips train.zip and test.zip in given folder"""
    train_archive_file = "{}.zip".format("train")
    train_archive_path = os.path.join(data_folder, train_archive_file)
    test_archive_file = "{}.zip".format("test")
    test_archive_path = os.path.join(data_folder, test_archive_file)

    train_path = os.path.join(data_folder, "train/")
    test_path = os.path.join(data_folder, "test/")

    if not os.path.exists(train_path):
        zip_ref = zipfile.ZipFile(train_archive_path, 'r')
        zip_ref.extractall(data_folder)
        zip_ref.close()
        print('Train Data extracted.')
    if not os.path.exists(test_path):
        zip_ref = zipfile.ZipFile(test_archive_path, 'r')
        zip_ref.extractall(data_folder)
        zip_ref.close()
        print('Test Data extracted.')

def copy_imgs(dir, dir_img, dir_masks):
    for filename in os.listdir(dir):
        if filename.endswith("mask.tif"):
            _image = os.path.join(dir, filename)
            assert os.path.isfile(_image)
            dst = os.path.join(dir_masks, filename)
            copyfile(_image, dst)
        elif filename.endswith(".tif"):
            _image = os.path.join(dir, filename)
            dst = os.path.join(dir_img, filename)
            copyfile(_image, dst)
        else:
            continue

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)


def to_cropped_imgs(ids, dir, suffix, scale, rgb=False):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        if rgb:
            im = resize_and_crop(Image.open(dir + id + suffix).convert('RGB'), scale=scale)
        else:
            im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.tif', scale, rgb=True)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.tif', scale)
    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.tif')
    im = im.convert('RGB')
    mask = Image.open(dir_mask + id + '_mask.tif')
    return np.array(im), np.array(mask)
