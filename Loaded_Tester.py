from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import json
import datetime
from datetime import datetime
import numpy as np
import os
import argparse
from ResNet18Model import ResNet18
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import L2DeepFoolAttack
from statistics import mean

parser = argparse.ArgumentParser(description="CIFAR10 Training")
parser.add_argument('--lr', type=float, default = 0.1, metavar='LR', help='learning rate')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed(default: 1')
parser.add_argument('--beta', default=0.1, type=float, help='loss weight for proximity')
parser.add_argument('--norm-type', default='batch', help='batch,layer, or instance')
parser.add_argument('--par-grad-mult', default=10.0, type=float, help='boost image gradients if desired')
parser.add_argument('--par-grad-clip', default=0.01, type=float, help='max magnitude per update for proto image updates')
parser.add_argument('--channel-norm', default=1, type=int, help='normalize each channel by training set mean and std')
parser.add_argument('--model-dir', default='../ProtoRuns')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', help='input batch size for testing (default: 128)')
parser.add_argument('--dataset', default='CIFAR10', help="Dataset being used")
parser.add_argument('--assessments', nargs='+', default=[],help='list of strings showing which assessments to make')
parser.add_argument('--image-step', default=0.0, type=float, help='for image training, number of decimals to round to')
parser.add_argument('--restart-epoch', default=100, type=int, help='epoch to restart from')
parser.add_argument('--schedule', nargs='+', type=float, default=[0.1, 0.25, 0.4, 0.6, 0.7, 0.8, 1.0], help='training points to consider')
parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W')
parser.add_argument('--anneal', default="cosine", help='type of LR schedule (cosine)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N', help='how many epochs to wait before logging training status')
parser.add_argument('--model-scale', default=1.0, type=float, help='width scale of network off of baseline resnet18')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--total-runs', type=int, default=5, help='How many instantiations of prototype images')


args = parser.parse_args()



def get_datetime():
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H_%M_%S")
    return dt_string
date_time = get_datetime()
targ = "/home/lab/csc-reu/yxp441/YashProto/PrototypeEvaluation_ResNet18_CIFAR10"
plottarg = "/home/lab/csc-reu/yxp441/YashProto/PrototypeEvaluation_ResNet18_CIFAR10/metric_plots"
dir_suffix = args.model_dir

model_dir = os.path.join(targ, dir_suffix)
full_dir_plot = os.path.join(plottarg, dir_suffix)
saved_model_path = os.path.join(model_dir,'../Trained_Model.pt')
saved_protos_path = os.path.join(model_dir,'../Saved_Protos.pt')

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(full_dir_plot):
    os.makedirs(full_dir_plot)

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

torch.manual_seed(args.seed)

kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}
torch.cuda.empty_cache()

print ("cuda: ", use_cuda)


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2471, 0.2435, 0.2616]

    transformDict = {}
    transformDict['norm'] = transforms.Compose([transforms.Normalize(MEAN, STD)])

    nclass = 10
    nchannels = 3
    H, W = 32, 32

    model = ResNet18(nclass=nclass, scale=args.model_scale, channels=nchannels, **kwargsUser).to(device)
    model_saved = torch.load(f"{saved_model_path}/Saved_Model")
    model.load_state_dict(model_saved)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False




