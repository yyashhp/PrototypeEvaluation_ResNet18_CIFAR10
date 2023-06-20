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
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed(default: 1')
parser.add_argument('--model-dir', default='../ProtoRuns')
parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
parser.add_argument('--model-scale', default=1.0, type=float, help='width scale of network off of baseline resnet18')
parser.add_argument('--total-runs', default=5, type=int, help='proto instantiations')
parser.add_argument('--dataset', default=1, type=int, help='know which dataset is loaded')

args = parser.parse_args()

kwargsUser = {}

def get_datetime():
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H_%M_%S")
    return dt_string
date_time = get_datetime()
targ = "/home/lab/csc-reu/yxp441/YashProto/PrototypeEvaluation_ResNet18_CIFAR10"
plottarg = "/home/lab/csc-reu/yxp441/YashProto/PrototypeEvaluation_ResNet18_CIFAR10/metric_plots"
dir_suffix = args.model_dir

model_dir = os.path.join(targ, dir_suffix)
full_dir_plot = plottarg
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

    model = ResNet18(nclass=nclass, scale=args.model_scale, channels=nchannels, **kwargsUser).to(device)
    model_saved = torch.load(f"{saved_model_path}/Saved_Model")
    model.load_state_dict(model_saved)
    model.multi_out = 1
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    par_image_tensors = torch.load(f"{saved_protos_path}/Saved_Protos")
    #
    # for run in range(args.total_runs):
    #     _par_image_copy = par_image_tensors[run].clone().detach().requires_grad_(False).to(device)
    #     _par_image_norm = transformDict['norm'](_par_image_copy)
    #     L2_img, logits_img = model(_par_image_norm)
    #     preds = logits_img.max(1, keepdim=True)[1]
    #     probs = F.softmax(logits_img)
    #    # print(f"Preds for run {run}:\t {preds}\n")
    #    # print(f"Probs for run {run}:\t {probs}\n")
    #
    # cos_matrices = []
    # for proto in par_image_tensors:
    #     par_tensors_norm = transformDict['norm'](proto.clone())
    #     latent_p, logits_p = model(par_tensors_norm)
    #
    #     # compute cos similarity matrix
    #
    #     cos_mat_latent_temp = torch.zeros(nclass, nclass, dtype=torch.float)
    #     cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
    #
    #     for i in range(len(latent_p)):
    #         for q in range(len(latent_p)):
    #             if i != q:
    #                 cos_mat_latent_temp[i, q] = 1 - cos_sim(latent_p[i].view(-1), latent_p[q].view(-1))
    #               #  print(cos_mat_latent_temp[i, q])
    #     cos_matrices.append(cos_mat_latent_temp.clone())
    #
    # cos_mat_std, cos_mat_mean = torch.std_mean(torch.stack(cos_matrices, dim=0), dim=0)
    # CS_mean = (torch.sum(cos_mat_mean.clone()))/((nclass*nclass)-nclass)
    # with open('{}/CS_stats_{}.txt'.format(model_dir, date_time), 'a') as f:
    #     f.write("\n")
    #     f.write(
    #         f"Each Protos CS_Diff_Mean, {cos_mat_mean.clone()} \t Overall CS_Diff_Mean {CS_mean.clone()}")
    #     f.write("\n")
    # f.close()
    # print(f"CS_DIFF_MEAN: {CS_mean}")
    #
    # L2_latent_means = []
    # CS_latent_means = []
    #
    # for proto in par_image_tensors:
    #     proto_copy = proto.clone()
    #     with torch.no_grad():
    #         proto_copy_norm = transformDict['norm'](proto_copy)
    #         latent_onehot, logit_onehot = model(proto_copy_norm)
    #     model.eval()
    #     model.multi_out = 0
    #     attack = L2DeepFoolAttack(overshoot=0.02)
    #     preprocessing = dict(mean=MEAN, std=STD, axis=-3)
    #     fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    #
    #     print('Computing DF L2_stats')
    #
    #     raw, new_proto, is_adv = attack(fmodel, proto_copy.clone(),
    #                                     torch.arange(nclass, dtype=torch.long, device=device), epsilons=10.0)
    #
    #     model.multi_out = 1
    #
    #     with torch.no_grad():
    #         new_proto_norm = transformDict['norm'](new_proto.clone())
    #         latent_adv, logits_adv = model(new_proto_norm)
    #     L2_df_latent = torch.linalg.norm((latent_adv - latent_onehot).view(nclass, -1), dim=1)
    #     CS_df_latent = 1 - F.cosine_similarity(latent_adv.view(nclass, -1), latent_onehot.view(nclass, -1))
    #
    #     latent_df_std, latent_df_mean = torch.std_mean(L2_df_latent)
    #     L2_latent_means.append(latent_df_mean.clone())
    #     CS_latent_means.append(torch.mean(CS_df_latent).clone())
    #     with open('{}/trained_Adv_stats_{}.txt'.format(model_dir, date_time), 'a') as f:
    #         f.write("\n")
    #         f.write(f"Batch's L2 diffs : {L2_df_latent} \n CS latent diffs: {CS_df_latent} \n L2 diff mean: {latent_df_mean} \n CS Latent diff Mean: {torch.mean(CS_df_latent).clone()}  ")
    #         f.write("\n")
    #     f.close()
    #
    #
    # L2_cum_latent_std, L2_cum_latent_mean = torch.std_mean(torch.stack(L2_latent_means, dim=0), dim=0)
    # L2_cum_latent_mean = (L2_cum_latent_mean.clone())
    #
    # CS_latent_std, CS_latent_mean = torch.std_mean(torch.stack(CS_latent_means, dim=0), dim=0)
    # CS_adv_latent = CS_latent_mean.clone()
    # with open('{}/Adv_stats_{}.txt'.format(model_dir, date_time), 'a') as f:
    #     f.write("\n")
    #     f.write(f"L2_diff latent overall mean: {L2_cum_latent_mean.clone()} \t CS_diff latent overall mean {CS_adv_latent.clone()}")
    # f.close()

    for proto in par_image_tensors:
        proto_copy = proto.clone()
        with torch.no_grad():
            proto_copy_norm = transformDict['norm'](proto_copy)
            latent_proto, logits_proto = model(proto_copy_norm)
            preds = logits_proto.max(1, keepdim=True)[1]
            probs = F.softmax(logits_proto)
        for i in range(len(proto)):
            for j in range(len(proto)):
                if i != j:
                    start_pred = preds[j]
                    end_pred = preds[i]
                    start_probs = probs[j]
                    end_probs = probs[i]
                    print(f"Start and End Preds and Probs: {start_pred}, {end_pred}, {start_probs}, {end_probs}")
                    start_image = proto_copy[j].clone().detach().requires_grad_(False).to(device)
                    target_class_image = proto_copy[i].clone().detach().requires_grad_(False).to(device)
                    print(f"Start and target class shapes: {start_image.shape}, {target_class_image.shape}")
                    for alpha in range(1,20):
                        adj_alpha = 1/(alpha)







if __name__ == '__main__':
    main()
