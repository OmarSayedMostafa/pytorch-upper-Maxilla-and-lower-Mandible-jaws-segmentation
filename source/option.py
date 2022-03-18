import argparse

def get_args():
    parser = argparse.ArgumentParser(description='VIPriors Segmentation baseline training script')

    # model architecture & checkpoint
    parser.add_argument('--model', metavar='[UNet, DeepLabv3_resnet50, DeepLabv3_resnet101]', 
                        default='UNet', type=str, help='model')
    parser.add_argument('--save_path', metavar='path/to/save_results', default='./baseline_run',
                        type=str, help='path to results saved')
    parser.add_argument('--weights', metavar='path/to/checkpoint', default=None,
                        type=str, help='resume training from checkpoint')
    parser.add_argument('--norm', metavar='[batch, instance, evo, group]', default='batch',
                        type=str, help='replace batch norm with other norm')

    # data loading
    parser.add_argument('--dataset_path', metavar='path/to/datset/root', default='./dataset',
                        type=str, help='path to dataset (ends with /minicity)')
    parser.add_argument('--pin_memory', metavar='[True,False]', default=True,
                        type=bool, help='pin memory on GPU')
    parser.add_argument('--num_workers', metavar='8', default=8, type=int,
                        help='number of dataloader workers')

    # data augmentation hyper-parameters
    parser.add_argument('--hflip', metavar='[True,False]', default=True,
                        type=float, help='data augmentation: random horizontal flip')
    parser.add_argument('--vflip', metavar='[True,False]', default=True,
                        type=float, help='data augmentation: random virtical flip')
    
    parser.add_argument('--scale', metavar='[True,False]', default=True,
                        type=float, help='data augmentation: random scale')

    parser.add_argument('--rotate', metavar='[True,False]', default=True,
                        type=float, help='data augmentation: random rotate')
                    

    parser.add_argument('--save_test_imgs', metavar='[True,False]', default=True,
                        type=float, help='save test prediction images')

    parser.add_argument('--save_val_imgs', metavar='[True,False]', default=True,
                        type=float, help='save val prediction images of last epoch')

    parser.add_argument('--train_size', default=[128, 128], nargs='+', type=int, help='image size during training')
    parser.add_argument('--test_size', default=[128, 128], nargs='+', type=int, help='image size during test')
    
    parser.add_argument('--dataset_mean', metavar='0.0',
                        default=0.0, type=float,
                        help='mean for normalization')

    parser.add_argument('--dataset_std', metavar='1.0',
                        default=1.0, type=float,
                        help='std for normalization')

    # training hyper-parameters
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--lr_init', metavar='1e-2', default=1e-2, type=float,
                        help='initial learning rate')
    parser.add_argument('--lr_momentum', metavar='0.9', default=0.9, type=float,
                        help='momentum for SGD optimizer')
    parser.add_argument('--lr_weight_decay', metavar='1e-4', default=1e-4, type=float,
                        help='weight decay for SGD optimizer')
    parser.add_argument('--epochs', metavar='200', default=200, type=int,
                        help='number of training epochs')
    parser.add_argument('--seed', metavar='42', default=None, type=int,
                        help='random seed to use')
    parser.add_argument('--loss', metavar='[ce, weighted_ce, focal]', default='ce',
                        type=str, help='loss criterion')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='initial learning rate')

    # inference options
    parser.add_argument('--predict', action='store_true')

    args = parser.parse_args("")
    return args