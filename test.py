# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, transforms
import os
import scipy.io
import yaml
from model import ft_net, PCB, PCB_test, ft_net_concat, ft_net_ori
# from model2 import ft_net_dg
from model_dg import ds_net
import json

device = torch.device("cuda:0, 1" if torch.cuda.is_available() else "cpu")

# fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='market1501/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='market1501_layer_4', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )

opt = parser.parse_args()


def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def main():

    ###load config###
    # load the training config
    config_path = os.path.join('./model',opt.name,'opts.yaml')
    with open(config_path, 'r') as stream:
            config = yaml.load(stream)
    opt.fp16 = False # config['fp16']
    opt.PCB = config['PCB']
    opt.use_dense = config['use_dense']

    str_ids = opt.gpu_ids.split(',')
    # which_epoch = opt.which_epoch
    name = opt.name
    test_dir = opt.test_dir

    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >=0:
            gpu_ids.append(id)

    # set gpu ids
    if len(gpu_ids)>0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    ######################################################################
    # Load Data
    # ---------
    #
    # We will use torchvision and torch.utils.data packages for loading the
    # data.
    #
    data_transforms = transforms.Compose([
            transforms.Resize((256,128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ############### Ten Crop
            #transforms.TenCrop(224),
            #transforms.Lambda(lambda crops: torch.stack(
             #   [transforms.ToTensor()(crop)
              #      for crop in crops]
               # )),
            #transforms.Lambda(lambda crops: torch.stack(
             #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
              #       for crop in crops]
              # ))
    ])

    if opt.PCB:
        data_transforms = transforms.Compose([
            transforms.Resize((384,192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    data_dir = test_dir
    if opt.multi:
        image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                 shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
    else:
        image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                      shuffle=False, num_workers=16) for x in
                       ['gallery', 'query']}
    use_gpu = torch.cuda.is_available()

    ######################################################################
    # Load model
    # ---------------------------
    def load_network(network):
        save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
        network.load_state_dict(torch.load(save_path))
        return network

    ######################################################################
    # Extract feature
    # ----------------------
    #
    # Extract feature from  a trained model.
    #
    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip

    def extract_feature(model, dataloaders):
        features = torch.FloatTensor()
        count = 0
        for data in dataloaders:
            img, label = data
            n, c, h, w = img.size()
            count += n
            print(count)
            if opt.use_dense:
                ff = torch.FloatTensor(n, 1024).zero_().to(device)
            else:
                ff = torch.FloatTensor(n, 512).zero_().to(device)
            if opt.PCB:
                ff = torch.FloatTensor(n, 2048, 6).zero_().to(device) # we have six parts
            for i in range(2):
                if(i==1):
                    img = fliplr(img)
                input_img = img.to(device)
                if opt.fp16:
                    input_img = input_img.half()
                _, _, outputs = model(input_img)
                ff += outputs
            # norm feature
            if opt.PCB:
                # feature size (n,2048,6)
                # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
                # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)
            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))

            ff = ff.data.cpu().float()
            features = torch.cat((features,ff), 0)
        return features

    def get_id(img_path):
        camera_id = []
        labels = []
        for path, v in img_path:
            #filename = path.split('/')[-1]
            filename = os.path.basename(path)
            label = filename[0:4]
            camera = filename.split('c')[1]
            if label[0:2]=='-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera[0]))
        return camera_id, labels

    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    if opt.multi:
        mquery_path = image_datasets['multi-query'].imgs
        mquery_cam, mquery_label = get_id(mquery_path)

    ######################################################################
    # Load Collected data Trained model
    print('-------test-----------')
    if opt.use_dense:
        model_structure = ft_net_dense(751)
    else:
        model_structure = ds_net(751)

    if opt.PCB:
        model_structure = PCB(751)

    if opt.fp16:
        model_structure = network_to_half(model_structure)

    model = load_network(model_structure)
    # model = model_structure

    # Remove the final fc layer and classifier layer

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.to(device)

    # Extract feature
    with torch.no_grad():
        gallery_feature = extract_feature(model,dataloaders['gallery'])
        query_feature = extract_feature(model,dataloaders['query'])
        if opt.multi:
            mquery_feature = extract_feature(model,dataloaders['multi-query'])

    # Save to Matlab for check
    result = {'gallery_f':gallery_feature.numpy(), 'gallery_label':gallery_label, 'gallery_cam':gallery_cam,
              'query_f':query_feature.numpy(), 'query_label':query_label, 'query_cam':query_cam}
    scipy.io.savemat('dsnet_test.mat', result)
    if opt.multi:
        result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
        scipy.io.savemat('multi_query.mat',result)


if __name__ == '__main__':
    main()