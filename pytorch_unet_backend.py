from __future__ import absolute_import
import os
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from protoseg.backends import AbstractBackend
from protoseg.trainer import Trainer

from tensorboardX import SummaryWriter

from .unet import UNet


class pytorch_unet_backend(AbstractBackend):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = None  # used for onnx export
    graph_exported = False

    def __init__(self):
        super(pytorch_unet_backend, self).__init__()

    def load_model(self, config, modelfile):
        model = UNet(n_channels=3, n_classes=config['classes']).to(self.device)
        if os.path.isfile(modelfile):
            print('loaded model from:', modelfile)
            model = torch.load(modelfile)
        model = torch.nn.DataParallel(
            model, device_ids=range(torch.cuda.device_count()))
        self.dummy_input = None
        self.graph_exported = False
        return model

    def save_model(self, model):
        torch.save(model.model,
                   model.modelfile)
        print('saved model to:', model.modelfile)

    def get_optimizer(self, name, parameters, config):
        optimizer = None
        if name == 'sgd':
            optimizer = torch.optim.SGD(parameters,
                                        lr=config['learn_rate'])
        elif name == 'adadelta':
            optimizer = torch.optim.Adadelta(parameters,
                                             lr=config['learn_rate'])
        elif name == 'adagrad':
            optimizer = torch.optim.Adagrad(
                parameters, lr=config['learn_rate'])
        elif name == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=config['learn_rate'])
        elif name == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                parameters, lr=config['learn_rate'])
        else:
            optimizer = torch.optim.SGD(parameters,
                                        lr=config['learn_rate'])
        return optimizer


    def init_trainer(self, trainer):
        trainer.loss_function = nn.CrossEntropyLoss().to(self.device)
        trainer.optimizer = torch.optim.SGD(trainer.model.model.parameters(), lr=trainer.config['learn_rate'],
                            momentum=0.9,
                            weight_decay=5e-4)
        trainer.model.model.train()

    def dataloader_format(self, img, mask=None):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = np.transpose(img, axes=[2, 0, 1])
        if mask is None:
            return img.astype(np.float32)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        mask[mask > 0] = 1  # binary mask
        img = img.astype(np.float32)
        mask = mask.astype(np.int64)
        return torch.from_numpy(img), torch.from_numpy(mask)

    def train_epoch(self, trainer):
        batch_size = trainer.config['batch_size']
        summarysteps = trainer.config['summarysteps']
        epoch_loss = 0

        dataloader = data.DataLoader(
            trainer.dataloader, batch_size=batch_size, num_workers=min(batch_size, 8), shuffle=True
        )

        for (images, labels) in tqdm(dataloader):
            trainer.global_step += 1
            trainer.model.model.train()
            if self.dummy_input is None:
                self.dummy_input = images.to(torch.device('cpu'))
            images = images.to(self.device)
            labels = labels.to(self.device)

            trainer.optimizer.zero_grad()
            outputs = trainer.model.model(images)

            loss = trainer.loss_function(input=outputs, target=labels)

            loss.backward()
            trainer.optimizer.step()
            trainer.loss += loss.item()

            if trainer.global_step % summarysteps == 0:
                print('{0:.4f} --- loss: {1:.6f}'.format(trainer.global_step *
                                                         batch_size / len(trainer.dataloader), loss.item()))
                if trainer.summarywriter:
                    trainer.summarywriter.add_scalar(
                        trainer.name+'loss', loss.item(), global_step=trainer.global_step)
                    trainer.summarywriter.add_image(
                        trainer.name+'image', images[0], global_step=trainer.global_step)
                    trainer.summarywriter.add_image(
                        trainer.name+'mask', labels[0], global_step=trainer.global_step)
                    pred = outputs.data.max(1)[1].cpu().numpy()
                    trainer.summarywriter.add_image(
                        trainer.name+'predicted', pred[0], global_step=trainer.global_step)
                    if not self.graph_exported:
                        try:
                            trainer.summarywriter.add_graph(
                                trainer.model.model, images)
                            self.graph_exported = True
                        except Exception as e:
                            print(e)

    def validate_epoch(self, trainer):
        batch_size = trainer.config['batch_size']
        dataloader = data.DataLoader(
            trainer.valdataloader, batch_size=batch_size, num_workers=min(batch_size, 8), shuffle=True
        )
        for i, (X_batch, y_batch) in tqdm(enumerate(dataloader)):
            prediction = self.batch_predict(trainer, X_batch)
            trainer.metric(
                prediction[0], y_batch[0].numpy(), prefix=trainer.name)
            if trainer.summarywriter:
                trainer.summarywriter.add_image(
                    trainer.name+"val_image", (X_batch[0]/255.0), global_step=trainer.epoch)
                trainer.summarywriter.add_image(
                    trainer.name+"val_mask", (y_batch[0]), global_step=trainer.epoch)
                trainer.summarywriter.add_image(
                    trainer.name+"val_predicted", (prediction[0]), global_step=trainer.epoch)

    def get_summary_writer(self, logdir='results/'):
        return SummaryWriter(log_dir=logdir)

    def predict(self, predictor, img):
        img_batch =torch.from_numpy(np.array([img]))
        return self.batch_predict(predictor, img_batch)

    def batch_predict(self, predictor, img_batch):
        model = predictor.model.model

        try:
            model = model.module
        except Exception:
            pass
        model.eval()
        images = img_batch.to(self.device)
        outputs = model(images)
        pred = outputs.data.max(1)[1].cpu().numpy()
        return pred