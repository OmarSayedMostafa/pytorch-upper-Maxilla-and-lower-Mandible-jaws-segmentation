import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.autograd import Variable

from learning.model import convert_bn_to_instancenorm, convert_bn_to_evonorm, convert_bn_to_groupnorm, DeepLabHead, UNet

import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import json
import glob
import os
import random

def load_jsonL_file(json_file_path):
    '''
    load .json and .jsonl files
    parameters : 
        json_file_path : (str) path of the file that end with .json or .jsonl to be loaded and returned as python dict
    return :
        json_dict : (dict) contains the json object
    '''
    with open(json_file_path) as f:
        if json_file_path[-1]=='l':
            print("\n[INFO] loading jsonl file from {}".format(json_file_path))
            json_dict = [json.loads(jline) for jline in tqdm(f.readlines())]
        else:
            json_dict = json.load(f)
            

    return json_dict


def save_json_file(json_file_path, json_object):
    '''
    save json dict in given file path
    parameters :
        json_file_path : (str) path including filename (e.g /home/user/file_name.json)
        json_object : (dict)
    '''
    with open(json_file_path, mode='w') as f:
        json.dump(json_object, f)


def plot_classes_balance(data_loader, data_name, include_bg=True):
    class_occurrance = defaultdict(int)
    for img_batch, lbl_batch in tqdm(data_loader):
        unique, counts = np.unique(lbl_batch, return_counts=True)
        for u, c in zip(unique, counts):
            class_occurrance[u]+=c

    if not include_bg:
        class_occurrance.setdefault(0, 0)
        del class_occurrance[0]
        
    classes , occ = list(dict(class_occurrance).keys()), list(dict(class_occurrance).values())
    plt.title(data_name)
    plt.xlabel('classes')
    plt.ylabel('occurrance')
    plt.bar(classes, occ)
    plt.show()


def plot_batch(batch):
    images, labels = batch
    fig,ax = plt.subplots(images.shape[0], 2, figsize=(8, 80))
    for i in range(images.shape[0]):
        ax[i,0].imshow(images[i].numpy().squeeze(), cmap='bone')
        ax[i,1].imshow(labels[i].numpy().squeeze(), cmap='gray')
    plt.show()



"""
====================
Data Loader Function
====================
"""
def merge_lists(dict_of_lists):
    merged_list = []
    for key in dict_of_lists:
        merged_list+=dict_of_lists[key]
    return merged_list

def get_dataloader(dataset, args):

    def test_trans(image, mask=None):
        # Resize, 1 for Image.LANCZOS
        image = TF.resize(image, args.test_size, interpolation=1) 
        # Normalize
        if args.normalize:
            image = TF.normalize(image, args.dataset_mean, args.dataset_std)
        
        # From PIL to Tensor
        image = TF.to_tensor(image)
        
        if mask:
            # Resize, 0 for Image.NEAREST
            mask = TF.resize(mask, args.test_size, interpolation=0) 
            mask = np.array(mask, np.uint8) # PIL Image to numpy array
            mask = torch.from_numpy(mask) # Numpy array to tensor
            return image, mask
        else:
            return image

    def train_trans(image, mask):
        image = TF.resize(image, args.train_size, interpolation=1) 
        mask = TF.resize(mask, args.train_size, interpolation=0) 


        phflip = np.random.randint(0,1) > 0.5
        pvflip = np.random.randint(0,1) > 0.5
        pscale = np.random.randint(0,1) > 0.5
        protate = np.random.randint(0,1) > 0.5

        if pscale and args.scale:
            # Random scaling
            scale_factor = np.random.uniform(0.5, 1.5)
            scaled_train_size = [int(element * scale_factor) for element in args.train_size]
            # Resize, 1 for Image.LANCZOS
            image = TF.resize(image, scaled_train_size, interpolation=1)
            # Resize, 0 for Image.NEAREST
            mask = TF.resize(mask, scaled_train_size, interpolation=0) 
        

        # H-flip
        if phflip == True and args.hflip == True:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # V-flip
        elif pvflip == True and args.vflip == True:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        if protate and args.rotate:
            rotate_angle = np.random.randint(10, 45)
            image = TF.rotate(image, angle=rotate_angle, center=image.size()//2, interpolation=1)
            mask = TF.rotate(mask, angle=rotate_angle, center=image.size()//2, interpolation=0)
        
        # Normalize
        if args.normalize:
            image = TF.normalize(image, args.dataset_mean, args.dataset_std)
        
        # From PIL to Tensor
        image = TF.to_tensor(image)
        
        # Convert ids to train_ids
        mask = np.array(mask, np.uint8) # PIL Image to numpy array

        n_channel_mask = np.zeros(shape=(mask.shape[0], mask.shape[1], args.lbl_channels), dtype=np.uint)
        n_channel_mask[np.where(mask==1)]=[0,1,0]
        n_channel_mask[np.where(mask==2)]=[0,0,2]

        mask = torch.from_numpy(mask) # Numpy array to tensor

            
        return image, mask


    train_data = {
        case: glob.glob(os.path.join(args.dataset_path, "train", case,"*.dicom.npy.gz"))
        for case in os.listdir(os.path.join(args.dataset_path, "train"))
    }
    test_data = glob.glob(os.path.join(args.dataset_path, "test/**/*.dicom.npy.gz"))

    validation_data = {}
    # sample from all cases
    for case in train_data:
        random.shuffle(train_data[case])
        assert len(train_data[case]) > 0
        validation_files_count = math.ceil(len(train_data[case]) * args.validation_ratio)
        validation_data[case] = train_data[case][:validation_files_count]
        train_data[case] = train_data[case][validation_files_count:]
    
    train_data = merge_lists(train_data)
    validation_data = merge_lists(validation_data)

    print(f"[INFO] found {len(train_data)} training examples, {len(validation_data)} validation example, {len(test_data)} test example")

    trainset = dataset(train_data, transforms=train_trans)
    valset = dataset(validation_data, transforms=test_trans)
    testset = dataset(test_data, transforms=test_trans)
    dataloaders = {}    
    dataloaders['train'] = torch.utils.data.DataLoader(trainset,
               batch_size=args.batch_size, shuffle=True,
               pin_memory=args.pin_memory, num_workers=args.num_workers)
    dataloaders['val'] = torch.utils.data.DataLoader(valset,
               batch_size=args.batch_size, shuffle=False,
               pin_memory=args.pin_memory, num_workers=args.num_workers)
    dataloaders['test'] = torch.utils.data.DataLoader(testset,
               batch_size=args.batch_size, shuffle=False,
               pin_memory=args.pin_memory, num_workers=args.num_workers)

    return dataloaders

"""
====================
Focal Loss
code reference: https://github.com/clcarwin/focal_loss_pytorch
====================
"""

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


"""
====================
Loss Function
====================
"""

def get_lossfunc(dataset, args):
    # Define loss, optimizer and scheduler
    if args.loss == 'ce':
        if args.model == 'DeepLabv3_resnet50':
            criterion = nn.CrossEntropyLoss(ignore_index=dataset.voidClass)
        else:
            criterion = nn.CrossEntropyLoss()
    elif args.loss == 'weighted_ce':
        # Class-Weighted loss
        class_weight = [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
        class_weight.append(0) #for void-class
        class_weight = torch.FloatTensor(class_weight).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=dataset.voidClass)
    elif args.loss =='focal':
        criterion = FocalLoss(gamma=args.focal_gamma)
    else:
        raise NameError('Loss is not defined!')

    return criterion


"""
====================
Model Architecture
====================
"""

def get_model(dataset, args):
    if args.model == 'UNet':
        """ U-Net baeline """
        model = UNet(1, len(dataset.validClasses), batchnorm=True)
    elif args.model == 'DeepLabv3_resnet50':
        """ DeepLab v3 ResNet50 """
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
        model.classifier = DeepLabHead(2048, len(dataset.validClasses))
    elif args.model == 'DeepLabv3_resnet101':
        """ DeepLab v3 ResNet101 """
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False)
        model.classifier = DeepLabHead(2048, len(dataset.validClasses))
    else:
        raise NameError('Model is not defined!')
    
    # Normalization Layer
    if args.norm == 'instance':
        convert_bn_to_instancenorm(model)
    elif args.norm == 'evo':
        convert_bn_to_evonorm(model)
    elif args.norm == 'group':
        convert_bn_to_groupnorm(model, num_groups=32)
    elif args.norm == 'batch':
        pass
    else:
        raise NameError('Normalization is not defined!')

    return model

"""
====================
random bbox function for cutmix
====================
"""

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

"""
====================
Custom copyblob function for copyblob data augmentation
====================
"""

def copyblob(src_img, src_mask, dst_img, dst_mask, src_class, dst_class):
    mask_hist_src, _ = np.histogram(src_mask.numpy().ravel(), len(MiniCity.validClasses)-1, [0, len(MiniCity.validClasses)-1])
    mask_hist_dst, _ = np.histogram(dst_mask.numpy().ravel(), len(MiniCity.validClasses)-1, [0, len(MiniCity.validClasses)-1])

    if mask_hist_src[src_class] != 0 and mask_hist_dst[dst_class] != 0:
        """ copy src blob and paste to any dst blob"""
        mask_y, mask_x = src_mask.size()
        """ get src object's min index"""
        src_idx = np.where(src_mask==src_class)
        
        src_idx_sum = list(src_idx[0][i] + src_idx[1][i] for i in range(len(src_idx[0])))
        src_idx_sum_min_idx = np.argmin(src_idx_sum)        
        src_idx_min = src_idx[0][src_idx_sum_min_idx], src_idx[1][src_idx_sum_min_idx]
        
        """ get dst object's random index"""
        dst_idx = np.where(dst_mask==dst_class)
        rand_idx = np.random.randint(len(dst_idx[0]))
        target_pos = dst_idx[0][rand_idx], dst_idx[1][rand_idx] 
        
        src_dst_offset = tuple(map(lambda x, y: x - y, src_idx_min, target_pos))
        dst_idx = tuple(map(lambda x, y: x - y, src_idx, src_dst_offset))
        
        for i in range(len(dst_idx[0])):
            dst_idx[0][i] = (min(dst_idx[0][i], mask_y-1))
        for i in range(len(dst_idx[1])):
            dst_idx[1][i] = (min(dst_idx[1][i], mask_x-1))
        
        dst_mask[dst_idx] = src_class
        dst_img[:, dst_idx[0], dst_idx[1]] = src_img[:, src_idx[0], src_idx[1]]