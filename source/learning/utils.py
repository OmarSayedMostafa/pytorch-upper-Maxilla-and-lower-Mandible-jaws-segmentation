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
from learning.losses import FocalLoss, DiceLoss, CE_DiceLoss, LovaszSoftmax



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

def get_dataloader(dataset, args, as_one_batch=False):

    def test_trans(image, mask=None):
        # Resize, 1 for Image.LANCZOS
        image = TF.resize(image, args.test_size, interpolation=1) 

        # From PIL to Tensor
        image = TF.to_tensor(image)
        
        # Normalize
        if args.normalize:
            image = TF.normalize(image, args.dataset_mean, args.dataset_std)
        
        if mask:
            # Resize, 0 for Image.NEAREST
            mask = TF.resize(mask, args.test_size, interpolation=0) 

            # convert 2d label to 3d label n channel classes
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
            scale_factor = np.random.uniform(0.75, 1.25)
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
            rotate_angle = np.random.randint(0, 45)
            image = TF.rotate(image, angle=rotate_angle, center=image.size()//2, interpolation=1)
            mask = TF.rotate(mask, angle=rotate_angle, center=image.size()//2, interpolation=0)
        
        # From PIL to Tensor
        image = TF.to_tensor(image)
        
        # Normalize
        if args.normalize:
            image = TF.normalize(image, args.dataset_mean, args.dataset_std)
        
        # convert 2d label to 3d label n channel classes
        mask = np.array(mask, np.uint8) # PIL Image to numpy array

        mask = torch.from_numpy(mask) # Numpy array to tensor
            
        return image, mask

    if as_one_batch:
        args.validation_ratio=0

    test_data = glob.glob(os.path.join(args.dataset_path, "test/**/*.dicom.npy.gz"))

    if args.dtype == 'splits':
        train_data = glob.glob(os.path.join(args.dataset_path, "train/**/*.dicom.npy.gz"))
        validation_data = glob.glob(os.path.join(args.dataset_path, "val/**/*.dicom.npy.gz"))
    
    elif args.dtype == 'random':
        train_data = glob.glob(os.path.join(args.dataset_path, "train", "**","*.dicom.npy.gz"))
        random.shuffle(train_data)
        assert len(train_data) > 0
        validation_files_count = math.ceil(len(train_data) * args.validation_ratio)
        validation_data = train_data[:validation_files_count]
        train_data = train_data[validation_files_count:]
    
    elif args.dtype == 'stratified':
        train_data = {
            case: glob.glob(os.path.join(args.dataset_path, "train", case,"*.dicom.npy.gz"))
            for case in os.listdir(os.path.join(args.dataset_path, "train"))
        }
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

    print(f"[INFO] found {len(train_data)} training examples, {len(validation_data)} validation example, {len(test_data)} test example @", args.dataset_path)

    trainset = dataset(train_data, transforms=train_trans)
    valset = dataset(validation_data, transforms=test_trans)
    testset = dataset(test_data, transforms=test_trans)

    if as_one_batch:
        args.batch_size = len(trainset)
        
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
        class_weight = args.class_weights
        class_weight = torch.FloatTensor(class_weight).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    elif args.loss =='diceloss':
        criterion = DiceLoss()
    elif args.loss =='ce_diceloss':
        criterion = CE_DiceLoss()
    elif args.loss =='weighted_ce_diceloss':
        class_weight = args.class_weights
        class_weight = torch.FloatTensor(class_weight).cuda()
        criterion = CE_DiceLoss(weight=class_weight)
    elif args.loss =='focal':
        class_weight = args.class_weights
        class_weight = torch.FloatTensor(class_weight).cuda()
        criterion = FocalLoss(gamma=args.focal_gamma,alpha=class_weight) 
    elif args.loss == 'tversky_loss':
        class_weight = args.class_weights
        class_weight = torch.FloatTensor(class_weight).cuda()
        criterion = LovaszSoftmax()
    
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
        model = UNet(1, len(dataset.validClasses), batchnorm=True, dropout_p=args.dropout_p)
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