from __future__ import print_function

import statistics

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
import matplotlib
import matplotlib.pyplot as plt
import csv

parser = argparse.ArgumentParser(description="CIFAR100 Training")
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed(default: 1')
parser.add_argument('--model-dir', default='../BoundaryRuns')
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
saved_model_path = os.path.join(model_dir,'../Saved_Models')
saved_protos_path = os.path.join(model_dir,'../Saved_FINAL_Protos.pt')

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(full_dir_plot):
    os.makedirs(full_dir_plot)

if not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)
if not os.path.exists(saved_protos_path):
    os.makedirs(saved_protos_path)

saved_boundaries_path = os.path.join(model_dir,'../Boundary_Data')
if not os.path.exists(saved_boundaries_path):
    os.makedirs(saved_boundaries_path)


use_cuda = not args.no_cuda and torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

torch.manual_seed(args.seed)

kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}
torch.cuda.empty_cache()

print ("cuda: ", use_cuda)

MEAN = [0.5] * 3
STD = [0.5] * 3
data_schedule = [0.25,0.4,0.6,0.7,0.8,0.9,1.0]
train_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, padding=4)])
train_transform_tensor = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.RandomCrop(32, padding=4)])
gen_transform_test = transforms.Compose([transforms.ToTensor()])
transformDict = {}

transformDict['basic'] = train_transform
transformDict['flipcrop'] = train_transform_tensor
transformDict['norm'] = transforms.Compose([transforms.Normalize(MEAN, STD)])
transformDict['mean'] = MEAN
transformDict['std'] = STD


def train_image_no_data(args, model, device, epoch, par_images, targets, transformDict):
    print ("training images against one hot encodings")
    model.eval()

    for batch_idx in range(1):
        _par_images_opt = par_images.clone().detach().requires_grad_(True).to(device)

        _par_images_opt_norm = transformDict['norm'](_par_images_opt)

        L2_img, logits_img = model(_par_images_opt_norm)

        loss = F.cross_entropy(logits_img, targets, reduction='none')
        loss.backward(gradient=torch.ones_like(loss))

        with torch.no_grad():
            gradients_unscaled = _par_images_opt.grad.clone()
            grad_mag = gradients_unscaled.view(gradients_unscaled.shape[0], -1).norm(2, dim=-1)
            #for grad in range(len(grad_mag)):
            #    gradd = grad_mag[grad]
            #    if gradd == 0:
            #        grad_mag[grad] = torch.mean(grad_mag)
            #print(f"Grad_Mag:{grad_mag}")
            image_grads = 0.1 * gradients_unscaled / grad_mag.view(-1, 1, 1, 1)
           # image_gradients = torch.nan_to_num(image_grads)
            #print(f"Printing image gradients here: {image_gradients}")
            if torch.mean(loss) > 1e-7:
                par_images.add_(-image_grads)
            par_images.clamp_(0.0, 1.0)

            _par_images_opt.grad.zero_()
    if batch_idx % 20 == 0:
        print('Train Epoch: {}\t BatchID: {}\t Loss {:.6f}'.format(epoch, batch_idx, torch.mean(loss).item()))

    with torch.no_grad():
        _par_images_final = par_images.clone().detach().requires_grad_(False).to(device)
        _par_images_final_norm = transformDict['norm'](_par_images_final)
        L2_img, logits_img = model(_par_images_final_norm)
        pred = logits_img.max(1, keepdim=True)[1]
        probs = F.softmax(logits_img)

    return loss, pred, probs




def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2471, 0.2435, 0.2616]

    transformDict = {}
    transformDict['norm'] = transforms.Compose([transforms.Normalize(MEAN, STD)])

    nclass = 100
    nchannels = 3
    cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
   # model = ResNet18(nclass=nclass, scale=args.model_scale, channels=nchannels, **kwargsUser).to(device)
   # model_saved = torch.load(f"{saved_model_path}/6_Saved_Model_with_1.0_Data_0621_16_53_47", map_location=device)
  #  model.load_state_dict(model_saved)
  #  model.multi_out = 1
  #  model.eval()

    final_comb_cs_diffs = []
    final_comb_l2_diffs = []
    final_ind_cs_diffs = []
    final_ind_l2_diffs = []
    final_comb_boundaries_avg = []
    final_comb_alphas_avg = []
    final_comb_cum_alphas_avg = []
    final_comb_trained_cs_diffs = [[] for _ in range(args.total_runs)]
    final_ind_trained_cs_diffs = [[] for _ in range(args.total_runs)]
    final_comb_trained_l2_diffs = []
    final_ind_trained_l2_diffs = []
    final_ind_trained_cs_diffs_std = [[] for _ in range(args.total_runs)]
    final_comb_trained_cols_cs_diffs = [[] for _ in range(args.total_runs)]
    final_comb_trained_cs_std = [[] for _ in range(args.total_runs)]
    final_comb_trained_col_cs_std = [[] for _ in range(args.total_runs)]
    final_ind_trained_col_cs_diffs = [[] for _ in range(args.total_runs)]
    final_ind_trained_cs_col_stds = [[] for _ in range(args.total_runs)]
    final_ind_interrow_diffs  = [[] for _ in range(args.total_runs)]
    final_ind_intercol_diffs= [[] for _ in range(args.total_runs)]
    final_ind_interrow_std= [[] for _ in range(args.total_runs)]
    final_ind_intercol_std= [[] for _ in range(args.total_runs)]
    final_interrow_diffs = [[] for _ in range(args.total_runs)]
    final_intercol_diffs = [[] for _ in range(args.total_runs)]
    final_interrow_std = [[] for _ in range(args.total_runs)]
    final_intercol_std = [[] for _ in range(args.total_runs)]
    col_maxes = [[] for _ in range(args.total_runs)]
    row_maxes = [[] for _ in range(args.total_runs)]
    row_mins = [[] for _ in range(args.total_runs)]
    col_mins = [[] for _ in range(args.total_runs)]
    row_high_outliers_total = [[] for _ in range(args.total_runs)]
    row_low_outliers_total = [[] for _ in range(args.total_runs)]
    col_high_outliers_total = [[] for _ in range(args.total_runs)]
    col_low_outliers_total = [[] for _ in range(args.total_runs)]
    batch_diff_std = []
    batch_diff_col_std = []
    mispredictions = []
    col_sorted_matrix = [[] for _ in range(args.total_runs)]
    row_sorted_matrix = [[] for _ in range(args.total_runs)]
    inter_row_image_diff = []
    end_logits = torch.zeros(nclass, nclass, dtype=torch.float)
    trained_boundary_sets = []
    stacked_sets_latent_boundaries = []

    stacked_trained_l2 = []
    stacked_sets_trained_boundaries = []
    col_quartiles_saved = [[] for _ in range(args.total_runs)]
    row_quartiles_saved = [[] for _ in range(args.total_runs)]
    interrow_quartiles_saved = [[] for _ in range(args.total_runs)]
    intercol_quartiles_saved = [[] for _ in range(args.total_runs)]
    saved_preds = [[] for _ in range(args.total_runs)]
    iterations_max = 0
    for j in range(len(data_schedule)):
        model = ResNet18(nclass=nclass, scale=args.model_scale, channels=nchannels, **kwargsUser).to(device)
        model_saved = torch.load(f"{saved_model_path}/{j}_Saved_Model_with_{data_schedule[j]}_CIFAR100_Data_0621_13_24_49", map_location=device)
        model.load_state_dict(model_saved)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        par_image_tensors_loaded = torch.load(f"{saved_protos_path}/CIFAR_100_Final_Saved_Protos_3_SPLIT_{j}", map_location=device)
        par_image_tensors = [set.clone() for set in par_image_tensors_loaded]
        iterations_needed = torch.zeros(nclass, nclass, dtype=torch.float)
        last_loss_save = torch.zeros(nclass, nclass, dtype=torch.float)
        col_quartiles = torch.zeros(nclass, 7, dtype=torch.float)
        row_quartiles = torch.zeros(nclass, 7, dtype = torch.float)
        intercol_quartiles = torch.zeros(nclass, 7, dtype=torch.float)
        interrow_quartiles = torch.zeros(nclass, 7, dtype=torch.float)
      #  boundary_images = torch.load(f"{saved_boundaries_path}/{data_schedule[j]}_Boundary_Images")
      #  boundary_latents = torch.load(f"{saved_boundaries_path}/{data_schedule[j]}_Boundary_Latent")
      ##  print(f"Sizes of boundary images loaded: {boundary_images.shape}")
      #  print(f"Sizes of boundary latent loaded: {boundary_latents.shape}")


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
    #     cos_mat_latent_temp = torch.zeros(nclass, nclass, dtype=torch.float, device = device)
    #
    #
    #     for i in range(len(latent_p)):
    #         for q in range(len(latent_p)):
    #             if i != q:
    #                 cos_mat_latent_temp[i, q] = cos_sim(latent_p[i].view(-1), latent_p[q].view(-1))
    #               #  print(cos_mat_latent_temp[i, q])
    #     cos_matrices.append(cos_mat_latent_temp.clone())
    #
    # cos_mat_std, cos_mat_mean = torch.std_mean(torch.stack(cos_matrices, dim=0), dim=0)
    ## CS_mean = (torch.sum(cos_mat_mean.clone()))/((nclass*nclass)-nclass)
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
    #
    #
    #     final_boundaries_list = []
    #     final_alphas_list = []
    #     final_cs_diffs = []
    #     final_l2_diffs = []
    #
    #     model.eval()
    #     for proto in par_image_tensors:
    #         batch_cs_diff = []
    #         batch_l2_diff = []
    #         alphas_list = []
    #         boundaries_list = []
    #         logits_list = []
    #         proto_copy = proto.clone()
    #         cos_latent_temp = torch.zeros(nclass, nclass, dtype=torch.float)
    #         #L2_latent_temp = torch.zeros(nclass, nclass, dtype=torch.float)
    #         with torch.no_grad():
    #             model.multi_out = 1
    #             proto_copy_norm = transformDict['norm'](proto_copy)
    #             latent_proto, logits_proto = model(proto_copy_norm)
    #             preds = logits_proto.max(1, keepdim=True)[1]
    #             probs = F.softmax(logits_proto)
    #             print(f"Proto preds: {preds}")
    #         for i in range(len(proto)):
    #             proto_boundaries = []
    #             proto_alphas = []
    #             boundary_latent = []
    #             cs_diff = []
    #             l2_diff = []
    #             for k in range(len(proto)):
    #                 if i == k:
    #                     proto_boundaries.append((torch.zeros([3, 32, 32], device=device)))
    #                 elif i != k:
    #                     start_pred = preds[k]
    #                     end_pred = preds[i]
    #                     start_probs = probs[k]
    #                     end_probs = probs[i]
    #                     start_image = proto_copy[k].clone().detach().requires_grad_(True).to(device)
    #                     #start_image = torch.unsqueeze(start_image, dim=0)
    #                     target_class_image = proto_copy[i].clone().detach().requires_grad_(True).to(device)
    #                    # target_class_image = torch.unsqueeze(target_class_image, dim=0)
    #                     print(f"Starting Pred: {start_pred}, target pred: {end_pred}\n")
    #                     prev = start_image
    #                     for alpha in range(1, 21):
    #                         adj_alpha = alpha * 0.05
    #                         tester = torch.zeros(*(list(start_image.shape)), device=device)
    #                         tester = torch.add(tester, start_image, alpha=(1-adj_alpha))
    #                         tester = torch.add(tester, target_class_image, alpha=adj_alpha)
    #                         tester_shaped = torch.unsqueeze(tester, dim=0)
    #                         #print(f"Tester Tensor: {tester_shaped}")
    #                         with torch.no_grad():
    #                             tester_norm = transformDict['norm'](tester_shaped.clone())
    #                        #     print(f"Tester_norm shape {tester_norm.shape}")
    #                             latent_tester, logits_tester = model(tester_norm)
    #                             preds_tester = logits_tester.max(1, keepdim=True)[1]
    #                             probs_tester = F.softmax(logits_tester)
    #                    #     print(f"Preds:Tester!: {preds_tester}")
    #                         if preds_tester == end_pred:
    #                             boundary = torch.zeros(*(list(tester.shape)), device=device)
    #                             boundary = torch.add(boundary, prev, alpha=0.5)
    #                             boundary = torch.add(boundary, tester, alpha=0.5)
    #                             boundary_shaped = torch.unsqueeze(boundary, dim=0)
    #                             print(
    #                             f"Boundary shape needed to go from proto {k} to proto {i} is {(1 - adj_alpha) * 100} percent proto {j} and {adj_alpha * 100} percent proto {i} \n")
    #
    #                             proto_boundaries.append(boundary.clone())
    #                             proto_alphas.append(adj_alpha)
    #                             with torch.no_grad():
    #                                 norm_boundary = transformDict['norm'](boundary_shaped.clone())
    #                                 boundary_latent, boundary_logits = model(norm_boundary)
    #                             cos_latent_temp[i][k] = (cos_sim(torch.squeeze(boundary_latent,dim = 0).view(-1),latent_proto[i].view(-1) ))
    #                             print(f"CS Diff as {k} goes to {i}: {cos_sim(torch.squeeze(boundary_latent, dim=0).view(-1),latent_proto[i].view(-1) )}\m ")
    #
    #                         # boundary_reshaped = torch.reshape(boundary.clone(), (3,1024))
    #
    #                             l2_diff.append(torch.mean(torch.linalg.norm((torch.squeeze(boundary_latent.clone(), dim=0) - latent_proto[i]), dim=0)))
    #                             print(f"l2-val as {k} goes to {i}: {torch.mean(torch.linalg.norm((torch.squeeze(boundary_latent.clone(), dim=0) - latent_proto[i]), dim=0))}\n" )
    #                             break
    #                         else:
    #                             prev = tester
    #                             assert alpha != 21
    #             boundaries_list.append(torch.stack(proto_boundaries, dim=0))
    #             alphas_list.append(proto_alphas)
    #             batch_l2_diff.append(torch.stack(l2_diff, dim=0))
    #             #batch_cs_diff.append(cs_diff)
    #         final_boundaries_list.append(torch.stack(boundaries_list, dim=0))
    #         final_alphas_list.append(alphas_list)
    #         final_cs_diffs.append(cos_latent_temp.clone())
    #         final_l2_diffs.append(torch.stack(batch_l2_diff, dim=0))
    #
    #     batch_cs = torch.mean(torch.stack(final_cs_diffs, dim=0), dim=0)
    #     print(f"Shape of batch_cs, {batch_cs.shape}")
    #     batch_cum_cs = torch.mean(batch_cs, dim=0)
    #     cum_cs_avg = 1 - torch.mean(batch_cum_cs)
    #     final_comb_cs_diffs.append(cum_cs_avg)
    #     final_ind_cs_diffs.append([1-val for val in batch_cum_cs])
    #
    #     batch_l2 = torch.mean(torch.stack(final_l2_diffs, dim=0), dim=0)
    #     batch_cum_l2 = torch.mean(batch_l2, dim=0)
    #     final_comb_l2_diffs.append(torch.mean(batch_cum_l2))
    #     final_ind_l2_diffs.append(batch_cum_l2)
    #
    #
    #     final_alphas = np.array(final_alphas_list)
    #     alpha_means = np.mean(final_alphas, axis = 0)
    #     alpha_means = np.mean(alpha_means, axis = 0)
    #     alpha_means = np.round(alpha_means, 2)
    #     mean_alpha_mean = np.mean(alpha_means)
    #     alpha_means = alpha_means.tolist()
    #     final_comb_alphas_avg.append(alpha_means)
    #     final_comb_cum_alphas_avg.append(mean_alpha_mean)
    #     # for i in range(nclass):
    #     #     final_alphas.append(mean(
    #     #         [mean(final_alphas_list[0][i]), mean(final_alphas_list[1][i]), mean(final_alphas_list[2][i]),
    #     #         mean(final_alphas_list[3][i]), mean(final_alphas_list[4][i])]))
    #     final_boundaries_avg = torch.stack(final_boundaries_list, dim=0)
    #     final_boundaries_avg = torch.squeeze(final_boundaries_avg)
    #     final_split_comb_boundaries_avg = torch.mean(final_boundaries_avg, dim=0)
    #     final_comb_boundaries_avg.append(torch.squeeze(final_split_comb_boundaries_avg, dim=0))
    #     print(f"average alphas list at split {j}: {alpha_means}\n")
    #     with open('{}/Line_Stats{}.txt'.format(saved_boundaries_path, date_time), 'a') as f:
    #         f.write("\n")
    #         f.write(
    #             f"Split {j} Average Alphas : {alpha_means}\n  average L2s: {batch_cum_l2}\n average CS_diff: {[1-val for val in batch_cum_cs]}\n,\
    #              Cumulative L2: {torch.mean(batch_cum_l2.clone())}\n Cumulative CS_diff: {cum_cs_avg}")
    #         f.write("\n")
    #     f.close()
        #torch.save(final_comb_boundaries_avg, f"{saved_boundaries_path}/Final_Combined_Linewise_Boundaries_{date_time}.pt")

    #     Boundary_L2_Diffs = []
    #     proto_index = 0
    #     for proto in par_image_tensors:
    #         proto_clone = proto.clone()
    #         # print(f"Proto clone shape\t {proto_clone[0].shape}\n")
    #         # print(f"Boundary shape\t {final_boundaries_avg[0][2][5].shape}\n")
    #         Batch_Boundary_Diffs = torch.zeros(nclass, nclass, dtype=torch.float, device=device)
    #         for i in range(nclass):
    #             for k in range(nclass):
    #                 if i != k:
    #                     boundary_reshaped = torch.reshape(final_boundaries_avg[j][proto_index][i][k].clone(), (3, 1024))
    #                     proto_reshaped = torch.reshape(proto_clone[i], (3, 1024))
    #                     Batch_Boundary_Diffs[i][j] = torch.mean(torch.linalg.matrix_norm(boundary_reshaped - proto_reshaped))
    #                     print(
    #                         f"L2 Diff in batch {proto_index} between proto {i} and boundary with {j} is {Batch_Boundary_Diffs[i][j]} \n")
    #         Boundary_L2_Diffs.append(Batch_Boundary_Diffs.clone())
    #         proto_index += 1
    #     L2_diff_std, L2_diff_mean = torch.std_mean(torch.stack(Boundary_L2_Diffs, dim=0), dim=0)
    #     tot_L2_diff_mean = torch.mean(L2_diff_mean.clone(), dim=0)
    #     print(f"Array of average L2_diff per class WRT each boundary: {L2_diff_mean}\n")
    #     print(f"Cumulative mean of L2 per proto: {tot_L2_diff_mean}")
    #     print(f"Cumulative mean of L2 overall: {torch.mean(tot_L2_diff_mean)}")
    #     with open('{}/Final_Linewise_Boundary_L2_{}.txt'.format(saved_boundaries_path, date_time), 'a') as f:
    #         f.write("\n")
    #         f.write(
    #             f"Array of average L2_diff per class WRT each boundary: {L2_diff_mean}\n \
    #                  Cumulative mean of L2 per proto: {tot_L2_diff_mean} \
    #                  Cumulative mean of L2 overall: {torch.mean(tot_L2_diff_mean)}")
    #         f.write("\n")
    #     f.close()
    #
        model.multi_out = 1
     #   stacked_sets_trained_boundaries = []
    #    stacked_sets_latent_boundaries = []
       # cos_trained_latent_matrices = []

       # cos_trained_latent_col_matrices = []
        interrow_values_matrices = []
        intercol_values_matrices = []
        #stacked_trained_l2 = []d
        set = -1
        # cos_trained_latent = torch.zeros(nclass, nclass, dtype=torch.float)
        # cos_trained_latent_col = torch.zeros(nclass, nclass, dtype=torch.float)
      #  for proto in par_image_tensors:


        for t in range(args.total_runs):
            proto = par_image_tensors[t].clone()
            boundary_images = torch.load(f"{saved_boundaries_path}/Cifar100_Batch{t}_{data_schedule[j]}_Boundaries_TrainedIms")
            boundary_latents = torch.load(f"{saved_boundaries_path}/Cifar100_Batch{t}_{data_schedule[j]}_Boundaries_Latents")
            print(f"Sizes of boundary images loaded: {len(boundary_images[0])}")
            print(f"Sizes of boundary latent loaded: {len(boundary_latents[0])}")
            set+=1
            cos_trained_latent = torch.zeros(nclass, nclass, dtype=torch.float)
            cos_trained_latent_col = torch.zeros(nclass, nclass, dtype=torch.float)
            interrow_values = torch.zeros([nclass, nclass, nclass], dtype=torch.float, device=device)
            intercol_values = torch.zeros([nclass, nclass, nclass], dtype=torch.float, device=device)
            preds_matrix = torch.zeros(nclass, nclass, dtype=torch.float, device=device)
            proto_clone = proto.clone()
        #    set_trained_boundaries = []
        #    batch_l2_trained_diff = []
        #    set_latent_boundaries = []
            with torch.no_grad():
                normed_protos = transformDict['norm'](proto_clone.clone())
                protos_latent, protos_logits = model(normed_protos)
            for i in range(nclass):
              #  trained_boundaries = []
              #  latents_boundaries = []
                target_proto = torch.tensor([i], device= device)
          #      l2_trained_diff = []
                for k in range(nclass):
                    if i == k:
                        for b in range(nclass):
                            if b == i:
                                interrow_values[i][k][b] = 0
                                intercol_values[i][k][b] = 0
                            else:
                                interrow_values[i][k][b] = cos_sim(protos_latent[i].clone(), boundary_latents[k][b].clone())
                                intercol_values[i][k][b] = cos_sim(protos_latent[i].clone(), boundary_latents[b][k].clone())
                   #     trained_boundaries.append((torch.zeros([3, 32, 32], device=device)))
                   #     latents_boundaries.append(torch.zeros(512, device=device))
                    elif i!=k:
                        trained_boundary = boundary_images[i][k].clone()
                        # iterations = 0
                        # epoch = 1
                        # last_loss = 100
                        start_proto = torch.unsqueeze(proto_clone[k].clone(), dim=0).clone()
                       # if i == 6:
                        start_proto_copy = start_proto.clone()
                        with torch.no_grad():
                            norm_trained_boundary = transformDict['norm'](torch.unsqueeze(trained_boundary, dim=0))
                            boundary_latent_dupe, boundary_logits = model(norm_trained_boundary)
                        preds = boundary_logits.max(1, keepdim=True)[1]
                        probs = F.softmax(boundary_logits)

                        #     normed_start = transformDict['norm'](start_proto_copy)
                        #     start_latent, start_logits = model(normed_start)
                        # start_preds_six = F.softmax(start_logits)
                        # starts_pred = start_logits.max(1, keepdim=True)[1]

                        # while last_loss>1e-2:
                        #     iterations += 1
                        #     last_loss, preds, probs = train_image_no_data(args, model=model, device=device,
                        #                                            epoch=epoch, par_images=start_proto,
                        #                                        targets=target_proto, transformDict=transformDict)
                        #     if iterations > 12500:
                        #         break
                        # if iterations_max < iterations < 12500:
                        #     iterations_max = iterations
                        print(f"Preds after {k} goes to {i}: {preds}\n")
                        if preds != i:
                            mispredictions.append([set, k, i, preds.item()])
                        #if i == 6:
                        preds_matrix[i][k] = preds
                        # with open('{}/Iterative_CIFAR100_split1_2_Until_Low_Loss_BOUNDARY_PROBS_{}.txt'.format(model_dir, date_time),
                        #           'a') as f:
                        #     f.write(
                        #         f"Going from {k} to {i}, batch {t},\t Iterations Needed: {iterations}\n\n")
                        #     # if iterations > 12500:
                        #     #     f.write(f"Iterations went over limit\n\n")
                        # f.close()
                        # iterations_needed[i][k] = iterations
                        model.eval()
                        # last_loss_save[i][k] = last_loss
                        # with torch.no_grad():
                        #     norm_trained_boundary = transformDict['norm'](start_proto.clone())
                        #     boundary_latent, boundary_logits = model(norm_trained_boundary)
                        start_proto_squeezed = torch.squeeze(start_proto.clone(), dim=0)
                    #    print(f"Boundary  shape: {start_proto_squeezed.shape}")
                        boundary_latent = boundary_latents[i][k].clone()
                    #    cos_trained_latent[i][k] = cos_sim(boundary_latent, protos_latent[i].clone())
                    #    cos_trained_latent_col[i][k] = cos_sim(boundary_latent, protos_latent[k].clone())
                        for b in range(nclass):
                            if b == k:
                                interrow_values[i][k][b] = 0
                                intercol_values[i][k][b] = cos_sim(boundary_latent, protos_latent[k].clone())
                            elif b == i:
                                intercol_values[i][k][b] = 0
                                interrow_values[i][k][b] = cos_sim(boundary_latent, protos_latent[i].clone())
                            else:
                                interrow_values[i][k][b] = cos_sim(boundary_latent, boundary_latents[i][b].clone())
                                intercol_values[i][k][b] = cos_sim(boundary_latent, boundary_latents[b][k].clone())
                        with open('{}/Batch0InterValsCalc_{}.txt'.format(model_dir, date_time),'a') as f:
                                f.write(
                                    f"Going from {k} to {i}, batch {t} \n\n")
                        f.close()

                       # trained_boundaries.append(start_proto_squeezed.clone())
                        # l2_trained_diff.append(torch.mean(torch.linalg.norm((boundary_latent.clone() - protos_latent[i].clone()), dim=0)))
                       # latents_boundaries.append(boundary_latent.clone())

            #    set_trained_boundaries.append(torch.stack(trained_boundaries, dim=0))
           #     set_latent_boundaries.append(torch.stack(latents_boundaries, dim=0))

            #    batch_l2_trained_diff.append(torch.stack(l2_trained_diff, dim=0))
          #  cos_trained_latent_matrices.append(cos_trained_latent.clone())
          #  cos_trained_latent_col_matrices.append(cos_trained_latent_col.clone())
            interrow_values_matrices.append(interrow_values.clone())
            intercol_values_matrices.append(intercol_values.clone())
        #    stacked_sets_trained_boundaries.append(torch.stack(set_trained_boundaries, dim=0))
            saved_preds[t].append(preds_matrix.clone())

          #  stacked_trained_l2.append(torch.stack(batch_l2_trained_diff, dim=0))

       #     torch.save(torch.stack(set_latent_boundaries, dim = 0), f"{saved_boundaries_path}/Final3_{data_schedule[j]}1_Boundaries_Latents_{date_time}.pt")
       #     torch.save(torch.stack(set_trained_boundaries, dim=0), f"{saved_boundaries_path}/Final3_{data_schedule[j]}1_Boundaries_Images_{date_time}.pt")

      #      stacked_sets_latent_boundaries.append(torch.stack(set_latent_boundaries, dim=0))
      # combined_boundary_images = torch.mean(torch.stack(stacked_sets_trained_boundaries,dim=0), dim=0)
       # trained_boundary_sets.append(torch.stack(stacked_sets_trained_boundaries, dim=0))
       # print(len(trained_boundary_sets))
       # combined_boundary_latent = torch.mean(torch.stack(stacked_sets_latent_boundaries, dim=0), dim=0)
      #  batch_trained_cs, batch_trained_std = torch.std_mean(torch.stack(cos_trained_latent_matrices, dim=0), dim=0)
      #  batch_trained_col_cs, batch_trained_col_std = torch.std_mean(torch.stack(cos_trained_latent_col_matrices, dim=0), dim=0)
       # batch_diff_std.append(batch_trained_std)
      #  batch_diff_col_std.append(batch_trained_col_std)
      #  batch_cum_trained_cs, batch_cum_trained_cs_std = torch.std_mean(batch_trained_cs, dim=0)
          #  batch_cum_trained_cs_std, batch_cum_trained_cs = torch.std_mean(cos_trained_latent.clone(), dim=1)
     #       mask = cos_trained_latent > 0
        #    col_mask = cos_trained_latent_col > 0
     #   batch_cum_trained_col_cs, batch_cum_trained_col_cs_std = torch.std_mean(batch_trained_col_cs, dim=1)
          #  batch_cum_trained_col_cs_std, batch_cum_trained_col_cs = torch.std_mean(cos_trained_latent_col.clone(),dim=0)
            batch_cum_trained_interrow_cs_std, batch_cum_trained_interrow_cs = torch.std_mean(interrow_values.clone(), dim=2)
            batch_cum_trained_intercol_cs_std, batch_cum_trained_intercol_cs = torch.std_mean(intercol_values.clone(), dim=2)

            print(f"Size of first means: {batch_cum_trained_intercol_cs.shape} \t {batch_cum_trained_interrow_cs.shape}")
            batch_cum_trained_interrow_cs = torch.mean(batch_cum_trained_interrow_cs, dim=1)
            batch_cum_trained_intercol_cs = torch.mean(batch_cum_trained_intercol_cs, dim=0)


        #    cum_trained_cs_avg = 1 - torch.mean(batch_cum_trained_cs)
      #      cum_trained_col_cs = 1 - torch.mean(batch_cum_trained_col_cs)
       #     std_list = []
       #     col_std_list = []
       #     for row in cos_trained_latent_matrices[t].clone():
       #         shortlist = []
       #         for val in row:
      #              if val>=1e-4:
     #                   shortlist.append(1-val)
     #           std_list.append(torch.stack(shortlist, dim=0))
     #       print(f'Lsize of the std_list: {torch.stack(std_list, dim=0).shape}')
     #       std_ave = torch.std(torch.stack(std_list, dim=0), dim=1)
      #      print(f"std_ave shape: {std_ave.shape}")
      #      mean_ave = torch.mean(torch.stack(std_list, dim=0), dim=1)
      #      print(f"Length of std_array {len(std_ave)}")
            inter_std_list = []
            intercol_std_list = []
            row_max = 0
            row_min = 100
            col_max = 0
            col_min = 100
            row_high_outliers = 0
            row_low_outliers = 0
            col_high_outliers = 0
            col_low_outliers = 0

            for row in interrow_values_matrices[t].clone():
                interrow_shortlist = []
                for deep in row:
                    #if deep[0].item() < 1e-8 and deep[1].item()< 1e-8 and torch.max(deep)[0].item() < 1e-8:
                    #    continue
                    for val in range(len(deep)):
                        print(f"Length of the deep: {len(deep)}")
                        if deep[val]>=1e-4:
                            interrow_shortlist.append(1-deep[val])
                            if 0.00001< deep[val]< row_min:
                                row_min = deep[val]
                            if 1 > deep[val]> row_max:
                                row_max = deep[val]
                inter_std_list.append(torch.stack(interrow_shortlist, dim=0))
        #    print(f'Lsize of the interrow_std_list: {torch.stack(inter_std_list, dim=0).shape}')
            row_maxes[t].append(1 - row_min)
            row_mins[t].append(1 - row_max)
            interrow_std_ave = torch.std(torch.stack(inter_std_list, dim=0), dim=1)
       #     interrow_std_ave = torch.mean(interrow_std_ave, dim=1)
        #    print(f"interrow_std_ave full shape: {inter_std_list.shape}")

        #    print(f"interrow_std_ave shape: {inter_std_list.shape}")

            # for row in cos_trained_latent_col_matrices[t].clone():
            #     col_shortlist = []
            #     for val in row:
            #         if val>=1e-4:
            #             col_shortlist.append(1-val)
            #         else:
            #             col_shortlist.append(torch.mean(row))
            #     col_std_list.append(torch.stack(col_shortlist, dim=0))
            # col_std_ave = torch.std(torch.stack(col_std_list, dim=0), dim=0)
            # print(f"col_std_ave shape: {col_std_ave.shape}")
            # cos_mean_ave = torch.mean(torch.stack(col_std_list, dim=0), dim=0)
            #
            # print(f"Length of col_std_array {len(col_std_ave)}")

            for row in torch.transpose(intercol_values_matrices[t].clone(), 0, 1).clone():
                intercol_shortlist = []
                for deep in row:
                    # if deep[0].item() < 1e-8 and deep[1].item()< 1e-8 and torch.max(deep)[0].item() < 1e-8:
                    #     continue
                    for val in range(len(deep)):
                        print(f"Length of the deep: {len(deep)}")

                        if deep[val]>=1e-4:
                            intercol_shortlist.append(1-deep[val])
                            if 0.0001< deep[val]< col_min:
                                col_min = deep[val]
                            if 1 > deep[val]> col_max:
                                col_max = deep[val]
                        else:
                            intercol_shortlist.append(torch.mean(deep))
                intercol_std_list.append(torch.stack(intercol_shortlist, dim=0))
            intercol_std_ave = torch.std(torch.stack(intercol_std_list, dim=0), dim=1)
            col_maxes[t].append(1 - col_min)
            col_mins[t].append(1 - col_max)

          #  intercol_std_ave = torch.mean(intercol_std_ave, dim=0)
       #     print(f"intercol_std_ave shape: {intercol_std_ave.shape}")
        #    cos_mean_ave = torch.mean(torch.stack(col_std_list, dim=0), dim=0)

       #     print(f"Length of intercol_std_array {len(intercol_std_ave)}")


            # final_comb_trained_col_cs_std[t].append(torch.mean(col_std_ave).item())
            # final_comb_trained_cs_std[t].append(torch.mean(std_ave).item())
            # final_ind_trained_col_cs_diffs[t].append([round(1 - ((val.item()* 100)/99), 4) for val in batch_cum_trained_col_cs])
            # final_ind_trained_cs_col_stds[t].append([round(val.item(), 4) for val in col_std_ave])
            # final_ind_trained_cs_diffs[t].append([round(1 - ((val.item() * 100)/99), 4) for val in batch_cum_trained_cs])
            # final_ind_trained_cs_diffs_std[t].append([round(val.item(), 4) for val in std_ave])
            row_std = torch.mean(interrow_std_ave).item()
            col_std = torch.mean(intercol_std_ave).item()
            row_mean = mean([round(1 - ((val.item() * 100) / 99), 4) for val in batch_cum_trained_interrow_cs])
            col_mean = mean([round(1 - ((val.item() * 100) / 99), 4) for val in batch_cum_trained_intercol_cs])
            print(f"Row means and stds and col ones: {row_mean} \t {row_std} \t {col_mean} \t {col_std}")



            final_ind_interrow_diffs[t].append(
                [round(1 - ((val.item() * 100) / 99), 4) for val in batch_cum_trained_interrow_cs])
            final_ind_intercol_diffs[t].append(
                [round(1 - ((val.item() * 100) / 99), 4) for val in batch_cum_trained_intercol_cs])
            final_ind_interrow_std[t].append([round(val.item(), 4) for val in interrow_std_ave])
            final_ind_intercol_std[t].append([round(val.item(), 4) for val in intercol_std_ave])
            final_interrow_diffs[t].append(row_mean)
            final_intercol_diffs[t].append(col_mean)
            final_interrow_std[t].append(row_std)
            final_intercol_std[t].append(col_std)

            for row in interrow_values_matrices[t].clone():
                for deep in row:
                    for val in range(len(deep)):
                        if deep[val] >= 1e-6:
                            if ((1 - deep[val].item()) - row_mean) / row_std >= 3:
                                row_high_outliers += 1
                            if ((1 - deep[val].item()) - row_mean) / row_std <= -3:
                                row_low_outliers += 1
            print(f'Lsize of the row high and low outlier list: {row_high_outliers} \t {row_low_outliers}')
            row_high_outliers_total[t].append(row_high_outliers)
            row_low_outliers_total[t].append(row_low_outliers)

            for row in torch.transpose(intercol_values_matrices[t].clone(), 0, 1).clone():
                for deep in row:
                    for val in range(len(deep)):
                        if deep[val] >= 1e-6:
                            if ((1 - deep[val].item()) - col_mean) / row_std >= 3:
                                col_high_outliers += 1
                            if ((1 - deep[val].item()) - col_mean) / row_std <= -3:
                                col_low_outliers += 1
            col_high_outliers_total[t].append(col_high_outliers)
            col_low_outliers_total[t].append(col_low_outliers)


        # batch_trained_l2 = torch.mean(torch.stack(stacked_trained_l2, dim=0), dim=0)
        # batch_cum_trained_l2 = torch.mean(batch_trained_l2, dim=0)
        # final_comb_trained_l2_diffs.append(torch.mean(batch_cum_trained_l2))
        # final_ind_trained_l2_diffs.append(batch_cum_trained_l2)

          #  final_comb_trained_cs_diffs[t].append(
        #    mean([round(1 - ((val.item() * 100) / 99), 4) for val in batch_cum_trained_cs]))
        #    final_comb_trained_cols_cs_diffs[t].append(
        #    mean([round(1 - ((val.item() * 100) / 99), 4) for val in batch_cum_trained_col_cs]))
        #    iterations_matrix[t].append(iterations_needed)

            line_index = 0
           #  for line in cos_trained_latent_matrices[t].clone():
           #      sorted_line = torch.sort(line.clone().detach(), descending=True)[0]
           #  #    print(f"length of line of row is {len(sorted_line)}")
           #      row_quartiles[line_index][0] = 1 - sorted_line[1]
           #      row_quartiles[line_index][1] = 1 - sorted_line[19]
           #      row_quartiles[line_index][2] = 1 - sorted_line[39]
           #      row_quartiles[line_index][3] = 1 - sorted_line[59]
           #      row_quartiles[line_index][4] = 1 - sorted_line[79]
           #      row_quartiles[line_index][5] = 1 - sorted_line[99]
           #      row_quartiles[line_index][6] = 1 - (torch.mean(sorted_line) * 100/99)
           #      line_index+=1
           #  row_quartiles_saved[t].append(row_quartiles.clone())
           #
           #  line_index = 0
           #
           #  for line in cos_trained_latent_col_matrices[t].clone():
           #      sorted_line = torch.sort(line.clone().detach(), descending=True)[0]
           # #     print(f"length of line of row is {len(sorted_line)}")
           #      col_quartiles[line_index][0] = 1 - sorted_line[1]
           #      col_quartiles[line_index][1] = 1 - sorted_line[19]
           #      col_quartiles[line_index][2] = 1 - sorted_line[39]
           #      col_quartiles[line_index][3] = 1 - sorted_line[59]
           #      col_quartiles[line_index][4] = 1 - sorted_line[79]
           #      col_quartiles[line_index][5] = 1 - sorted_line[99]
           #      col_quartiles[line_index][6] = 1 - (torch.mean(sorted_line) * 100/99)
           #      line_index+=1
           #  col_quartiles_saved[t].append(col_quartiles.clone())
           #  line_index = 0

            for line in interrow_values_matrices[t].clone():
                sorted_line = torch.sort(torch.flatten(line.clone().detach()), descending=True)[0]
                print(f"length of line of row is {len(sorted_line)}")
                interrow_quartiles[line_index][0] = 1 - sorted_line[0]
                interrow_quartiles[line_index][1] = 1 - sorted_line[1999]
                interrow_quartiles[line_index][2] = 1 - sorted_line[3999]
                interrow_quartiles[line_index][3] = 1 - sorted_line[5999]
                interrow_quartiles[line_index][4] = 1 - sorted_line[7999]
                interrow_quartiles[line_index][5] = 1 - sorted_line[9898]
                interrow_quartiles[line_index][6] = 1 - (torch.mean(sorted_line) * 100/99)
                line_index += 1

                row_sorted_matrix[t].append(sorted_line.tolist()[300:9700])
            interrow_quartiles_saved[t].append(interrow_quartiles.clone())
            line_index = 0

            for line in intercol_values_matrices[t].clone():
                sorted_line = torch.sort(torch.flatten(line.clone().detach()), descending=True)[0]
                #     print(f"length of line of row is {len(sorted_line)}")
                intercol_quartiles[line_index][0] = 1 - sorted_line[0]
                intercol_quartiles[line_index][1] = 1 - sorted_line[1999]
                intercol_quartiles[line_index][2] = 1 - sorted_line[3999]
                intercol_quartiles[line_index][3] = 1 - sorted_line[5999]
                intercol_quartiles[line_index][4] = 1 - sorted_line[7999]
                intercol_quartiles[line_index][5] = 1 - sorted_line[9898]
                intercol_quartiles[line_index][6] = 1 - (torch.mean(sorted_line) * 100/99)
                line_index += 1
                col_sorted_matrix[t].append(sorted_line.tolist()[300:9700])
            intercol_quartiles_saved[t].append(intercol_quartiles.clone())
            line_index = 0
            with open('{}/Iterative_CIFAR100_Row_Mat.txt'.format(model_dir), 'a') as f:

                # using csv.writer method from CSV package
                write = csv.writer(f)
                write.writerows(row_sorted_matrix)
            with open('{}/Iterative_CIFAR100_Col_Mat.txt'.format(model_dir), 'a') as f:
                write = csv.writer(f)
                write.writerows(col_sorted_matrix)

                # using csv.writer method from CSV package





        #class_diffs = []
        #diffs_check = []
        row_medians = []
        col_medians = []
        row_sorted_means = []
        col_sorted_means = []
        for t in range (args.total_runs):
            row_medians.append([row[49] for row in np.array(row_sorted_matrix[t]).T.tolist()])
            col_medians.append([col[49] for col in np.array(col_sorted_matrix[t]).T.tolist()])
            row_sorted_means.append(np.mean(row_sorted_matrix[t], axis=0))
            col_sorted_means.append(np.mean(col_sorted_matrix[t], axis=0))
        row_median = np.mean(row_medians, axis=0)
        col_median = np.mean(col_medians, axis=0)
        row_sorted_mean = np.mean(row_sorted_means, axis=0)
        col_sorted_mean = np.mean(col_sorted_means, axis=0)
        print(len(row_sorted_mean))
        print("LENGTH OF ROW SORTED MEAN")
        newer_x_axis = list(range(9400))
        plt.plot(newer_x_axis, row_sorted_mean, label="Mean")
        plt.plot(newer_x_axis, row_median, label="Median")

        plt.title(f'Averaged Intra-Class Cosine Similarity Mean and Median, with {data_schedule[j]} Data')
        plt.xlabel('Sorted Index')
        plt.ylabel('Cosine Similarity')
        plt.legend()
        plt.savefig(
            f"{model_dir}/../PrototypeEvaluation_ResNet18_CIFAR10/metric_plots/{date_time}_Intra_Class_{data_schedule[j]} Spread.png")
        plt.show()
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()

        plt.plot(newer_x_axis, col_sorted_mean, label="Mean")
        plt.plot(newer_x_axis, col_median, label="Median")

        plt.title(f'Averaged Inter-Class Cosine Similarity Mean and Median, with {data_schedule[j]} Data')
        plt.xlabel('Sorted Index')
        plt.ylabel('Cosine Similarity')
        plt.legend()
        plt.savefig(
            f"{model_dir}/../PrototypeEvaluation_ResNet18_CIFAR10/metric_plots/{date_time}_Inter_Class_{data_schedule[j]} Spread.png")
        plt.show()
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()



        # for t in range(1):
        #     matrix = stacked_sets_trained_boundaries[0].clone()
        #     protos_copy = par_image_tensors[0].clone()
        #     with torch.no_grad():
        #         norm_protos = transformDict['norm'](protos_copy)
        #         proto_latent, proto_logits = model(norm_protos)
        #     print(f"Length of matrix: {len(matrix)}")
        #     for i in range(len(matrix)):
        #         basefound = False
        #         inter_class_diffs = []
        #         inter_diffs_check = []
        #         base = 0
        #         for k in range(len(matrix)):
        #             if basefound is False and k != i:
        #                 base = matrix[i][k].clone()
        #                 basefound = True
        #                 based_index = k
        #                 inter_diffs_check.append(
        #                     1 - round(cos_sim(proto_latent[i], stacked_sets_latent_boundaries[0][i][k]).item(), 4))
        #             elif k != i:
        #                 inter_diff = cos_sim(matrix[i][k], matrix[i][based_index])
        #                 inter_latent = cos_sim(stacked_sets_latent_boundaries[0][i][k], stacked_sets_latent_boundaries[0][i][based_index])
        #                 inter_class_diffs.append([int(i), int(based_index), int(k), round(1 - round(torch.mean(inter_diff).item(),4), 4), round(1 - round(inter_latent.item(), 4), 4) ])
        #                 inter_diffs_check.append(1-round(cos_sim(proto_latent[i], stacked_sets_latent_boundaries[0][i][k]).item(), 4))
        #             else:
        #                 inter_class_diffs.append([0.0,0.0,0.0,0.0,0.0])
        #                 inter_diffs_check.append(0)
        #         class_diffs.append(inter_class_diffs)
        #         diffs_check.append(inter_diffs_check)





       # print(f"shape of combined_image and combined_latent: {combined_boundary_images.shape} /n {combined_boundary_latent.shape}")

        # with open('{}/Trained_Stats{}.txt'.format(saved_boundaries_path, date_time), 'a') as f:
        #     f.write("\n")
        #     f.write(
        #         f"Split {j} \n  average L2s: {batch_cum_trained_l2}\n average CS_diff: {[1-val for val in batch_cum_trained_cs]}\n,\
        #          Cumulative L2: {torch.mean(batch_cum_trained_l2.clone())}\n Cumulative CS_diff: {cum_trained_cs_avg}")
        #     f.write("\n")
        # f.close()

    # print(f"length of comb alphas : {len(final_comb_alphas_avg)}")
    # print(f"length of l2s : {len(final_ind_l2_diffs)}")
    # print(f"length of cs's : {len(final_ind_cs_diffs)}")

    # for j in range(len(data_schedule)):
    #     torch.save(stacked_sets_latent_boundaries[j],
    #                f"{saved_boundaries_path}/Final_{data_schedule[j]}_Boundaries_Latents_{date_time}.pt")
    #     torch.save(stacked_sets_trained_boundaries[j],
    #                f"{saved_boundaries_path}/Final_{data_schedule[j]}_Boundaries_TrainedIms_{date_time}.pt")

    with open('{}/Iterative_CIFAR100_Data_Collect_{}.txt'.format(model_dir, date_time), 'a') as f:
        #for i in range(6, len(data_schedule)):
        # f.write(f"\n Split: {data_schedule[i]} \t Alphas: {final_comb_alphas_avg[i]}  \t cumulative alpha: {final_comb_cum_alphas_avg[i]} \t CS_Line_Diffs:\
            #  {[val.item() for val in final_ind_cs_diffs[i]]} \
            # \n CS_Trained_Diffs{[val.item() for val in final_ind_trained_cs_diffs[i]]} \n Cumulative cs_Line diff: {final_comb_cs_diffs[i]} \n \
            #  Cumulative cs_trained_diff: {final_comb_trained_cs_diffs[i]} \n L2 line diffs(image): {final_ind_l2_diffs[i].tolist()} \
            #            \t  Cumulative L2 Diff: {final_comb_l2_diffs[i]}  \t \n \
            #                 L2_Trained_Diffs(latent): {final_ind_trained_l2_diffs[i].tolist()} \t Cumulative L2_Trained_Diff: {final_comb_trained_l2_diffs[i]}\n \n")

        # f.write(f"inter-batch std of row-wise CS diff: {batch_diff_std[0]} \n \
        #  inter-batch std of col-wise CS diff {batch_diff_col_std[0]} \n \
        #    row wise CS diffs: {final_ind_trained_cs_diffs[0]} \t \t \
        #     row wise CS stds: {final_ind_trained_cs_diffs_std[0]} \n \n  \
        #     column-wise CS diffs: {final_ind_trained_col_cs_diffs[0]} \t \t \
        #      column-wise CS stds: {final_ind_trained_cs_col_stds[0]} \n \n \
        #      \n \n cumulative row-wise CS diff: {final_comb_trained_cs_diffs[0]} \t \t \
        #       cumulative row-wise CS Std; {final_comb_trained_cs_std} \
        #        \n \n cumulative column-wise CS diff {final_comb_trained_cols_cs_diffs[0]} \
        #         \t \t cumulative column-wise CS Std: {final_comb_trained_col_cs_std[0]} \n \n \
        #          Mispredictions: {mispredictions}")
        # for i in range(len(data_schedule)):
        #     for t in range(args.total_runs):
        #         f.write(f" Data Split: {data_schedule[i]} \n  \
        #             Batch {t} \n \
        #             \n \n cumulative row-wise CS diff: \t {final_comb_trained_cs_diffs[t][i]} \n  \
        #              cumulative row-wise CS Std;\t {final_comb_trained_cs_std[t][i]} \
        #               \n  cumulative column-wise CS diff: \t {final_comb_trained_cols_cs_diffs[t][i]} \
        #                \n \n cumulative column-wise CS Std:\t {final_comb_trained_col_cs_std[t][i]} \n \n \
        #                 Mispredictions: \n{mispredictions} \n \n \
        #                    Batches Quartile Measures: Row-Wise: [min, 20, 40, 60, 80, max, average]: \n {row_quartiles_saved[t][i]} \
        #                      Batches Quartile Measures: Column-Wise: [min, 20, 40, 60, 80, max, average]: \n {col_quartiles_saved[t][i]} \n \n \n \n \n \
        #                         Interrow Cs diffs: \n {final_ind_interrow_diffs[t][i]} \t \t \n \
        #                         InterRow Stds: \n {final_ind_interrow_std[t][i]} \t \t \n \
        #                           InterCol Cs diffs: \n {final_ind_intercol_diffs[t][i]} \n \n  \
        #                           InterCol Stds: \n {final_ind_intercol_std[t][i]} \t \t \n \n \
        #                              InterRow Cs Quartiles: Row-Wise: [min, 20, 40, 60, 80, max, average]: \n {interrow_quartiles_saved[t][i]} \
        #                                 InterCol Cs Quartiles: Column-Wise: [min, 20, 40, 60, 80, max, average]: \n {intercol_quartiles_saved[t][i]} \
        #                                    InterRow Average CS: \n {final_interrow_diffs[t][i]} \n \n \
        #                                       InterCol Average CS: \n {final_intercol_diffs[t][i]}")
        #     f.write(f"\n Iterations max: {iterations_max}\n\n")

            for i in range(len(data_schedule)):
                for t in range(args.total_runs):
                    f.write(f" Data Split: {data_schedule[i]} \t  \
                               Batch {t} \t \
                                   Mispredictions: \t{mispredictions} \t \
                                           InterRow Stds: \t {final_interrow_std[t][i]} \t \
                                             InterCol Stds: \t {final_intercol_std[t][i]} \t \t \
                                                      InterRow Average CS: \t {final_interrow_diffs[t][i]} \t \
                                                       Max: {row_maxes[t][i]} \t  Min: {row_mins[t][i]} \t \
                                                         High Outliers: {row_high_outliers_total[t][i]} \t \
                                                           Low Outliers: {row_low_outliers_total[t][i]} \t \
                                                         InterCol Average CS: \t {final_intercol_diffs[t][i]} \
                                           Max: {col_maxes[t][i]} \t Min: {col_mins[t][i]} \t High Outliers: {col_high_outliers_total[t][i]} \
                                       Low outliers: {col_low_outliers_total[t][i]}")
                f.write(f"\tIterations max: {iterations_max}\n")


            #torch.set_printoptions(threshold=10000)
           # print(f"Predictions matrix: {saved_preds[t][i]}\n\n", file = f)
           # torch.set_printoptions(threshold=1000)


    f.close()
    for t in range(args.total_runs):
        #  plt.plot(data_schedule, final_comb_trained_cols_cs_diffs[t], label="column-wise cs diff")
        #   plt.plot(data_schedule, final_comb_trained_cs_diffs[t], label="row-wise cs diff")
        #   plt.plot(data_schedule, final_comb_trained_col_cs_std[t], label="column-wise std")
        #    plt.plot(data_schedule, final_comb_trained_cs_std[t], label="row-wise std")
        plt.plot(data_schedule, final_interrow_diffs[t], label="inter-row-wise diff")
        plt.plot(data_schedule, final_intercol_diffs[t], label="inter-col-wise diff")
        plt.legend()
        plt.savefig(
            f"{model_dir}/../PrototypeEvaluation_ResNet18_CIFAR10/metric_plots/{date_time}_CIFAR100_Saved_Data{t}.png")
        plt.show()
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()

    #  overall_row_cs_diffs = torch.mean(torch.Tensor(final_comb_trained_cs_diffs).clone(), dim=0)
    #  overall_col_cs_diffs = torch.mean(torch.Tensor(final_comb_trained_cols_cs_diffs).clone(), dim=0)
    overall_interrow_cs_diffs = torch.mean(torch.Tensor(final_interrow_diffs).clone(), dim=0)
    overall_intercol_cs_diffs = torch.mean(torch.Tensor(final_intercol_diffs).clone(), dim=0)
    #   overall_row_cs_stds = torch.mean(torch.Tensor(final_comb_trained_cs_std).clone(), dim=0)
    #  overall_col_cs_stds = torch.mean(torch.Tensor(final_comb_trained_col_cs_std).clone(), dim=0)
    overall_interrow_cs_stds = torch.mean(torch.Tensor(final_interrow_std).clone(), dim=0)
    overall_intercol_cs_stds = torch.mean(torch.Tensor(final_intercol_std).clone(), dim=0)

    #  plt.plot(data_schedule, overall_col_cs_diffs.tolist(), label="Origin-wise Inter-Class Dissimilarity")
    #  plt.plot(data_schedule, overall_row_cs_diffs.tolist(), label="Origin-wise Intra-Class Dissimilarity")
    plt.plot(data_schedule, overall_intercol_cs_diffs.tolist(), label="Inter-Class Dissimilarity")
    plt.plot(data_schedule, overall_interrow_cs_diffs.tolist(), label="Intra-Class Dissimilarity")
    plt.legend()
    plt.title('Cifar100 Inter-Class and Intra-Class Prototype Latent Vector Dissimilarity')
    plt.xlabel('Percentage of Data the Model was Trained on')
    plt.ylabel('Dissimilarity (1 - Cosine Similarity)')
    plt.savefig(f"{model_dir}/../PrototypeEvaluation_ResNet18_CIFAR10/metric_plots/{date_time}_Overall_Cif100Vals.png")
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    #  plt.plot(data_schedule, overall_col_cs_stds.tolist(), label="Origin-wise Inter-Class STD")
    #  plt.plot(data_schedule, overall_row_cs_stds.tolist(), label="Origin-wise Intra-Class STD")
    plt.plot(data_schedule, overall_intercol_cs_stds.tolist(), label="Inter-Class STD")
    plt.plot(data_schedule, overall_interrow_cs_stds.tolist(), label="Intra-Class STD")
    plt.legend()
    plt.title('Cifar100 STD of Prototype Latent Vector Dissimilarity')
    plt.xlabel('Percentage of Data the Model was Trained on')
    plt.ylabel('STD')
    plt.savefig(
        f"{model_dir}/../PrototypeEvaluation_ResNet18_CIFAR10/metric_plots/{date_time}_Overall_Cif100STDs.png")
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    new_x_axis = list(range(9400))
    plt.plot(new_x_axis, row_sorted_mean, label="Mean")
    plt.plot(new_x_axis, row_median, label="Median")

    plt.title('Averaged Intra-Class Cosine Similarity Mean and Median')
    plt.xlabel('Sorted Index')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.savefig(
        f"{model_dir}/../PrototypeEvaluation_ResNet18_CIFAR10/metric_plots/{date_time}_Intra_Class_Spread.png")
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    plt.plot(new_x_axis, col_sorted_mean, label="Mean")
    plt.plot(new_x_axis, col_median, label="Median")


    plt.title('Averaged Inter-Class Cosine Similarity Mean and Median')
    plt.xlabel('Sorted Index')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.savefig(
        f"{model_dir}/../PrototypeEvaluation_ResNet18_CIFAR10/metric_plots/{date_time}_Inter_Class_Spread.png")
    plt.show()
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


    overall_row_maxes = torch.max(torch.Tensor(row_maxes).clone(), dim=0)[0]
    overall_col_maxes = torch.max(torch.Tensor(col_maxes).clone(), dim=0)[0]
    overall_row_mins = torch.min(torch.Tensor(row_mins).clone(), dim=0)[0]
    overall_col_mins = torch.min(torch.Tensor(col_mins).clone(), dim=0)[0]
    overall_row_high_outliers = torch.mean(torch.Tensor(row_high_outliers_total).clone(), dim=0)
    overall_col_high_outliers = torch.mean(torch.Tensor(col_high_outliers_total).clone(), dim=0)
    overall_row_low_outliers = torch.mean(torch.Tensor(row_low_outliers_total).clone(), dim=0)
    overall_col_low_outliers = torch.mean(torch.Tensor(col_low_outliers_total).clone(), dim=0)
    print(overall_row_low_outliers.shape)

    with open('{}/quartile_data{}.txt'.format(model_dir, date_time), 'a') as f:
        for i in range(len(data_schedule)):
            f.write(
               f"{interrow_quartiles_saved[0]} \n \
                    {intercol_quartiles_saved[0]}")
    f.close()
    with open('{}/outlier_data{}.txt'.format(model_dir, date_time), 'a') as f:
        for i in range(len(data_schedule)):
            f.write(
                f"Data Split : {data_schedule[i]} \t Row Mean: {round(overall_interrow_cs_diffs[i].item(), 4)} \t Row_Std: {round(overall_interrow_cs_stds[i].item(), 4)} \
                  Max: {round(overall_row_maxes[i].item(), 4)} \t Min: {round(overall_row_mins[i].item(), 4)} \t # of high outliers: {round(overall_row_high_outliers[i].item(), 1)} \t # of low outliers: {round(overall_row_low_outliers[i].item(), 1)} \
                     \n Col mean: {round(overall_intercol_cs_diffs[i].item(), 4)} \t Col STD: {round(overall_intercol_cs_stds[i].item(), 4)} \t Max: {round(overall_col_maxes[i].item(), 4)} \t Min: {round(overall_col_mins[i].item(), 4)} \
                        # of high outliers: {round(overall_col_high_outliers[i].item(), 1)} \t # of low outliers: {round(overall_col_low_outliers[i].item(), 1)} \n")
    f.close()



if __name__ == '__main__':
    main()
