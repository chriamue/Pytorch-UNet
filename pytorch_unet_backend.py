from __future__ import absolute_import
import os
import numpy as np
import cv2

import torch
from torch import optim, nn
import torch.nn.functional as F
from .unet import UNet
from .utils import batch, hwc_to_chw

from protoseg.backends import AbstractBackend
from protoseg.trainer import Trainer

from tensorboardX import SummaryWriter


class pytorch_unet_backend(AbstractBackend):

    def __init__(self):
        AbstractBackend.__init__(self)

    def load_model(self, config, modelfile):
        model = UNet(n_channels=3, n_classes=1)
        if config['gpu']:
            model.cuda()
        if os.path.isfile(modelfile):
            print('loaded model from:', modelfile)
            model.load_state_dict(torch.load(modelfile))
        return model

    def save_model(self, model):
        torch.save(model.model.state_dict(),
                   model.modelfile)
        print('saved model to:', model.modelfile)

    def init_trainer(self, trainer):
        trainer.loss_function = nn.BCELoss()
        trainer.optimizer = optim.SGD(trainer.model.model.parameters(),
                          lr=trainer.config['learn_rate'],
                          momentum=0.9,
                          weight_decay=0.0005)

    def dataloader_format(self, img, mask=None):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = hwc_to_chw(img)
        if mask is None:
            return img.astype(np.float32)

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        return img.astype(np.float32), mask.astype(np.float32)

    def train_epoch(self, trainer):
        print('train on pytorch_unet_backend')
        batch_size = trainer.config['batch_size']
        summarysteps = trainer.config['summarysteps']

        epoch_loss = 0

        for i, b in enumerate(batch(trainer.dataloader, batch_size)):
            trainer.global_step += 1
            imgs = np.array([i[0] for i in b])
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if trainer.config['gpu']:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = trainer.model.model(imgs)
            masks_probs = torch.sigmoid(masks_pred)
            masks_probs_flat = masks_probs.view(-1)

            true_masks_flat = true_masks.view(-1)

            loss = trainer.loss_function(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            if i % summarysteps == 0:
                print('{0:.4f} --- loss: {1:.6f}'.format(i *
                                                         batch_size / len(trainer.dataloader), loss.item()))
                trainer.summarywriter.add_scalar('loss', loss.item(), global_step=trainer.global_step)
                trainer.summarywriter.add_image('image', imgs[0], global_step=trainer.global_step)
                trainer.summarywriter.add_image('mask', true_masks[0], global_step=trainer.global_step)
                trainer.summarywriter.add_image('predicted', masks_probs[0], global_step=trainer.global_step)

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()


    def validate_epoch(self, trainer):
        batch_size = trainer.config['batch_size']
        dataloader = trainer.valdataloader
        for i, (X_batch, y_batch) in enumerate(dataloader):
            #prediction = self.batch_predict(trainer, X_batch)
            trainer.summarywriter.add_image(
                "val_image", (X_batch[0]/255.0), global_step=trainer.epoch)
            trainer.summarywriter.add_image(
                "val_mask", (y_batch[0]/255.0), global_step=trainer.epoch)
            #trainer.summarywriter.add_image(
            #    "val_predicted", (prediction), global_step=trainer.epoch)

    def get_summary_writer(self, logdir='results/'):
        return SummaryWriter(log_dir=logdir)

    def predict(self, predictor, img):
        img_batch = batchify.Stack()([img])
        return self.batch_predict(predictor, img_batch)

    def batch_predict(self, predictor, img_batch):
        model = predictor.model.model
        try:
            model = model.module
        except Exception:
            pass
        with autograd.predict_mode():
            outputs = model(img_batch.as_in_context(self.ctx))
            output, _ = outputs
        predict = mxnet.nd.squeeze(
            mxnet.nd.argmax(output, 1)).asnumpy().clip(0, 1)
        return predict.astype(np.float32)
