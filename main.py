
import os
import sys
import yaml
import torch
from .train import train_net
from .unet import UNet
from .predict import predict_img
from .submit import submit



def main(run='run', results_dir="results/", batch_size=1, epochs=5, config={}, mode='--train'):
    net = UNet(n_channels=3, n_classes=1)
    if config['gpu']:
        net.cuda()
    if mode == '--train':
        train_net(net=net,
              epochs=epochs,
              batch_size=batch_size,
              lr=config['learn_rate'],
              gpu=config['gpu'],
              img_scale=0.5,
              results_dir=results_dir)
    if mode == '--submit':
        netfile = [x for x in os.listdir(results_dir) if x.endswith('.pth')]
        if len(netfile) > 0:
            netfile = sorted(netfile, key=lambda s: int(s[2:-4]))
            netfile.reverse()
            netfile = results_dir + netfile[0]
            print('Predicting on file: ' ,netfile)
            net.load_state_dict(torch.load(netfile))
            submit(net, gpu=config['gpu'], results_dir=results_dir)



if __name__ == "__main__":
    with open(sys.argv[1], 'r') as file:
        configs = yaml.load(file)
        for config in configs:
            c = configs[config]
            results_dir = "results/"+config+"/"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            with open(results_dir + 'config.yml', 'w') as outfile:
                yaml.dump({config: c}, outfile, default_flow_style=False)
            main(run=config, results_dir=results_dir, batch_size=c['batch_size'],
                 epochs=c['epochs'], config=c)
