import torch
import torch.nn.functional as F

from .dice_loss import dice_coeff


def eval_net(net, dataset, gpu=False, writer=None, epoch=0):
    """Evaluation without the densecrf with the dice coefficient"""
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]
        mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
        if not writer is None and i == 3:
            writer.add_image(tag='image', img_tensor=img, global_step=epoch)
            writer.add_image(
                tag='label', img_tensor=true_mask/255.0, global_step=epoch)
            writer.add_image(tag='prediction',
                             img_tensor=mask_pred, global_step=epoch)

        tot += dice_coeff(mask_pred, true_mask).item()
    return tot / i
