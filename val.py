import argparse
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations import Compose, RandomRotate90, Resize, Normalize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
import time
from archs import CSMamba-UNeXt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="CSMamba-UNeXt",
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Load config
    with open(f'models/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config:
        print(f'{key}: {config[key]}')
    print('-' * 20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])

    if config['arch'] == 'CSMamba-UNeXt':
        model = CSMamba-UNeXt(
            num_classes=config['num_classes'],
            img_size=config.get('img_size', 224),
            embed_dims=config.get('embed_dims', [128, 160, 256]),
            drop_path_rate=config.get('drop_path_rate', 0.1),
            drop_rate=config.get('drop_rate', 0.2)
        )
    else:
        model = archs.__dict__[config['arch']](
            config['num_classes'],
            config['input_channels'],
            config['deep_supervision']
        )

    model = model.cuda()

#Kvasir  CVC-ClinicDB  CVC-ColonDB ETIS-LaribPolypDB CVC-300

    # Data loading code
  #  img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = glob(os.path.join('inputs','CVC-ColonDB','images','*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

  #  _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    val_img_ids=img_ids
    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        RandomRotate90(),
        Resize(config['input_h'], config['input_w']),
        Normalize(),  # 使用直接导入的 Normalize()
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
       # img_dir=os.path.join('inputs', config['dataset'], 'images'),
      # mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_dir=os.path.join('inputs','CVC-ColonDB','images'),
        mask_dir=os.path.join('inputs','CVC-ColonDB','masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0
    #for c in range(config['num_classes']):
       #os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            print(type(input))
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)
            if isinstance(output, (list, tuple)):
                output = output[-1]  # 取主输出
            if output.shape[1] > 1 and config['num_classes'] == 1:
                output = output[:, :1]
            pred = torch.sigmoid(output)

            iou, dice = iou_score(pred, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            # 保存可视化用的二值图
            bin_out = (pred > 0.5).float().cpu().numpy()
            for i in range(len(bin_out)):
                cv2.imwrite(os.path.join('outputs', config['name'], "0", meta['img_id'][i] + '.jpg'),
                            (bin_out[i, 0] * 255).astype('uint8'))


            iou,dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0
              #resunet用下面的输出
            for i in range(len(output)):
               cv2.imwrite(os.path.join('outputs', config['name'], "0", meta['img_id'][i] + '.jpg'), (output[i, 0] * 255).astype('uint8'))


          #  for i in range(len(output)):
           #           for c in range(config['num_classes']):
            #              cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
             #                         (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

