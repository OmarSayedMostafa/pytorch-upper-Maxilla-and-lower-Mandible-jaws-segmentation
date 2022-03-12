# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import torch
import gzip
from torchvision import transforms as TF



class JawsDataset(torch.utils.data.Dataset):
    voidClass = -1
    validClasses = [0,1,2]
    classLabels = ['BG', 'Mandible', 'Maxilla']

    mask_colors = np.array([[0,0,0], [0,128,0],[0,0,128]])
    
    def __init__(self, dicom_file_list, transforms):
        self.dicom_file_list = dicom_file_list
        self.transforms = transforms
        self.toPil = TF.ToPILImage()

    def __len__(self):
        return len(self.dicom_file_list)

    def __getitem__(self, idx):
        dicom_path = self.dicom_file_list[idx]
        label_path = dicom_path.replace('.dicom.npy.gz', '.label.npy.gz')
        dicom_file = gzip.GzipFile(dicom_path, 'rb')
        dicom = np.load(dicom_file)
        label_file = gzip.GzipFile(label_path, 'rb')
        label = np.load(label_file)

        dicom, label = self.transforms(self.toPil(dicom), self.toPil(label))
        
        return dicom, label


