
from thop import profile




import argparse
import os
from glob import glob
from PIL import Image
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time

def parse_args():
    #实例化ArgumentParser
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="UNext0",
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])

    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', 'polyp', 'TestDataset', 'Kvasir', 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

  #  _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    val_img_ids = img_ids
    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
       # img_dir=os.path.join('inputs', config['dataset'], 'images'),
       # mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_dir=os.path.join('inputs', 'polyp', 'TestDataset', 'Kvasir', 'images'),
        mask_dir=os.path.join('inputs', 'polyp', 'TestDataset', 'Kvasir', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=3,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
#AverageMeter()该函数用来管理一些需要更新的变量，比如loss、accuracy、mAP等度量指标。直接用其定义一个待更新的变量。初始化时会调用AverageMeter函数中的reset方法
    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    gput = AverageMeter()
    cput = AverageMeter()

    count = 0
  #  for c in range(config['num_classes']):
   #     os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)



    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output

            flops, params = profile(model, (input,))
            print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))
            print(1)






            if count<=len(val_loader):

                start = time.time()
                if config['deep_supervision']:
                    output = model(input)[-1]
                else:
                    output = model(input)
                stop = time.time()

                gput.update(stop-start, input.size(0))

                start = time.time()
                model = model.cpu()
                input = input.cpu()
                output = model(input)
                stop = time.time()

                cput.update(stop-start, input.size(0))
                count=count+1

            iou,dice = iou_score(output, target)
            #调用AverageMeter()函数的update方法时会对相应的变量进行更新
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            # 下面一行自己加的 转Tensor
            output = torch.tensor(output)
            output = torch.sigmoid(output).cpu().numpy()


            for i in range(output.shape[0]):
                image_slice = output[i]  # 获取数组的一个切片
                image_slice = np.transpose(image_slice, (1, 2, 0))
                image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255
                image_slice=image_slice.astype(np.uint8)  # 调整通道顺序

                image = Image.fromarray(image_slice)  # 创建图像对象
                image.save(os.path.join('outputs', config['name'], meta['img_id'][i] +str(i)+ '.jpg'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    print('CPU: %.4f' %cput.avg)
    print('GPU: %.4f' %gput.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
