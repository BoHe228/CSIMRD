import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset_synapse import Synapse_dataset
from Model.Model import MRCFDNet
from utils.utils import test_single_volume
from torch.cuda.amp import autocast

print(torch.cuda.is_available())


parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str, default='results/model_pth/MFCA_Model_pth/best.pth')
parser.add_argument('--volume_path', type=str,
                    default='data/synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='data/synapse/lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=48,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', type=bool, default=False, help='whether to save results during inference')
parser.add_argument('--chunk', type=int,
                    default=0, help='Set whether chunking is required.')
parser.add_argument('--ks', type=int,
                    default=3, help='kernel sizes of UDCB')


parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

parser.add_argument('--snapshot_name', type=str, default='results/results/MFCA', help='saving prediction as nii!')
parser.add_argument('--test_save_dir', type=str, default='CCMIX_sfv2b_V1_', help='saving results')

parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--encoder', type=str,
                    default='pvt_v2_b2', help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
parser.add_argument('--seed', type=int, default=2222, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--kernel_sizes', type=int, nargs='+',
                    default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
parser.add_argument('--expansion_factor', type=int,
                    default=2, help='expansion factor in MSCB block')
parser.add_argument('--activation', type=str,
                    default='relu6', help='activation : relu6 or relu')
parser.add_argument('--no_dw_parallel', action='store_true', 
                    default=False, help='use this flag to disable depth-wise parallel convolutions')
parser.add_argument('--concatenation', action='store_true', 
                    default=False, help='use this flag to concatenate feature maps in MSDC block')
parser.add_argument('--no_pretrain', action='store_true', 
                    default=False, help='use this flag to turn off loading pretrained enocder weights')
parser.add_argument('--supervision', type=str,
                    default='mutation', help='loss supervision: mutation, deep_supervision or last_layer')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')

args = parser.parse_args()



def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1,pin_memory=True)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        image, label = image.to(device), label.to(device)
        
        
        with autocast():
          
            metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': 'data/synapse/test_vol_h5',
            'list_dir': 'data/synapse/lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    model = MRCFDNet(num_classes=args.num_classes, kernel_sizes=args.kernel_sizes,
                     expansion_factor=args.expansion_factor,
                     dw_parallel=not args.no_dw_parallel, activation=args.activation, encoder=args.encoder,
                     pretrain=not args.no_pretrain, chunk=args.chunk,ks = args.ks)

    print("Your model is loaded from {}.".format(args.test_path))
    checkpoint = torch.load(args.test_path)
    model.load_state_dict(checkpoint)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1 and args.n_gpu > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    print(device)
    model.to(device)
    
    snapshot_name = args.snapshot_name

    log_folder = 'results/test_logs' 
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))  


    if args.is_savenii:
        test_save_path = os.path.join(args.test_save_dir)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, model, test_save_path)


