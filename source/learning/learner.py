from helpers.helpers import AverageMeter, ProgressMeter, iouCalc, visim, vislbl
import torch
import torch.nn.functional as F
import os
import numpy as np
import time
from PIL import Image
import cv2
"""
=================
Routine functions
=================
"""

def train_epoch(dataloader, model, criterion, optimizer, lr_scheduler, epoch, void=-1, args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Accuracy', ':.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, loss_running, acc_running],
        prefix="Train, epoch: [{}]".format(epoch))
    
    # input resolution
    res = args.train_size[0]*args.train_size[1]
    
    # Set model in training mode
    model.train()
    
    end = time.time()
    
    with torch.set_grad_enabled(True):
        # Iterate over data.
        for epoch_step, (inputs, labels) in enumerate(dataloader):
            data_time.update(time.time()-end)
            
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(inputs)

            preds = torch.argmax(outputs, 1)
            # cross-entropy loss
            loss = criterion(outputs, labels) 

            # backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            bs = inputs.size(0) # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)
            corrects = torch.sum(preds == labels.data)
            nvoid = int((labels==void).sum())
            acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)
            
            # output training info
            progress.display(epoch_step)
            
            # Measure time
            batch_time.update(time.time() - end)
            end = time.time()

        # Reduce learning rate
        lr_scheduler.step(loss_running.avg)
        
    return loss_running.avg, acc_running.avg

    
def validate_epoch(dataloader, model, criterion, epoch, classLabels, validClasses, void=-1, maskColors=None, folder='baseline_run', args=None, mode='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_running = AverageMeter('Loss', ':.4e')
    acc_running = AverageMeter('Accuracy', ':.4e')
    iou = iouCalc(classLabels, validClasses, voidClass = void)
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, loss_running, acc_running],
        prefix=mode+", epoch: [{}]".format(epoch))
    
    # input resolution
    res = args.test_size[0]*args.test_size[1]
    
    # Set model in evaluation mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for epoch_step, (inputs, labels) in enumerate(dataloader):
            data_time.update(time.time()-end)
            
            inputs = inputs.float().cuda()
            labels = labels.long().cuda()
    
            # forward
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            bs = inputs.size(0) # current batch size
            loss = loss.item()
            loss_running.update(loss, bs)
            corrects = torch.sum(preds == labels.data)
            nvoid = int((labels==void).sum())
            acc = corrects.double()/(bs*res-nvoid) # correct/(batch_size*resolution-voids)
            acc_running.update(acc, bs)
            # Calculate IoU scores of current batch
            iou.evaluateBatch(preds, labels)
            
            # Save visualizations of first batch
            if args.save_val_imgs and maskColors is not None:
                for i in range(inputs.size(0)):
                    filename = (epoch_step*inputs.size(0))+i
                    # Only save inputs and labels once
                    if epoch == 0:
                        img = visim(inputs[i,:,:,:], args)
                        label = vislbl(labels[i,:,:], maskColors)
                        if len(img.shape) == 3:
                            cv2.imwrite(folder + '/images/{}/{}.png'.format(mode, filename),img[:,:,::-1])
                        else: 
                            cv2.imwrite(folder + '/images/{}/{}.png'.format(mode,filename),img)
                        cv2.imwrite(folder + '/images/{}/{}_gt.png'.format(mode,filename),label[:,:,::-1])
                    # Save predictions
                    pred = vislbl(preds[i,:,:], maskColors)
                    cv2.imwrite(folder + '/images/{}/{}_last_preds.png'.format(mode,filename),pred[:,:,::-1])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # print progress info
            progress.display(epoch_step)
        
        miou = iou.outputScores(mode, args)
        print('Accuracy      : {:5.3f}'.format(acc_running.avg))
        print('---------------------')

    return acc_running.avg, loss_running.avg, miou

def predict(dataloader, model, maskColors, folder='baseline_run', mode='val', args=None, Dataset=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time],
        prefix='Predict: ')
    
    # Set model in evaluation mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for epoch_step, batch in enumerate(dataloader):

            if len(batch) == 1:
                inputs = batch
            else:
                inputs, labels = batch

            data_time.update(time.time()-end)
            
            inputs = inputs.float().cuda()

            # forward
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
        
            # Save visualizations of first batch
            for i in range(inputs.size(0)):
                filename = epoch_step+(i*epoch_step)
                # Save input
                img = visim(inputs[i,:,:,:], args)
                img = Image.fromarray(img, 'RGB')
                img.save(folder + '/results_color_{}/{}_input.png'.format(mode, filename))
                # Save prediction with color labels
                pred = preds[i,:,:].cpu()
                pred_color = vislbl(pred, maskColors)
                pred_color = Image.fromarray(pred_color.astype('uint8'))
                pred_color.save(folder + '/results_color_{}/{}_prediction.png'.format(mode, filename))
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # print progress info
            progress.display(epoch_step)
