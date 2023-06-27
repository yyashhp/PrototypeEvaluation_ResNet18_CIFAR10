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
data_schedule = [0.25, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
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

    for batch_idx in range(100):
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

    nclass = 10
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
    final_comb_trained_cs_diffs = []
    final_ind_trained_cs_diffs = []
    final_comb_trained_l2_diffs = []
    final_ind_trained_l2_diffs = []
    final_ind_trained_cs_diffs_std = []
    final_comb_trained_cols_cs_diffs = []
    final_comb_trained_cs_std = []
    final_comb_trained_col_cs_std = []
    final_ind_trained_col_cs_diffs = []
    final_ind_trained_cs_col_stds = []
    batch_diff_std = []
    batch_diff_col_std = []
    mispredictions = []
    inter_row_image_diff = []
    cos_trained_latent = torch.zeros(nclass, nclass, dtype=torch.float)
    cos_trained_latent_col = torch.zeros(nclass, nclass, dtype=torch.float)
    last_loss_save = torch.zeros(nclass, nclass, dtype=torch.float)
    end_logits =  torch.zeros(nclass, nclass, dtype=torch.float)
    trained_boundary_sets = []
    stacked_sets_latent_boundaries = []

    stacked_trained_l2 = []
    stacked_sets_trained_boundaries = []

    for j in range(6, len(data_schedule)):
        model = ResNet18(nclass=nclass, scale=args.model_scale, channels=nchannels, **kwargsUser).to(device)
        model_saved = torch.load(f"{saved_model_path}/{j}_Saved_Model_with_{data_schedule[j]}_Data_0621_16_53_47", map_location=device)
        model.load_state_dict(model_saved)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        par_image_tensors = torch.load(f"{saved_protos_path}/Final_Saved_Protos_SPLIT_{j}", map_location=device)
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
        cos_trained_latent_matrices = []
        cos_trained_latent_col_matrices = []
        #stacked_trained_l2 = []
        set = -1
        # cos_trained_latent = torch.zeros(nclass, nclass, dtype=torch.float)
        # cos_trained_latent_col = torch.zeros(nclass, nclass, dtype=torch.float)
      #  for proto in par_image_tensors:
        for t in range(1):
            proto = par_image_tensors[0]
            set+=1
            # cos_trained_latent = torch.zeros(nclass, nclass, dtype=torch.float)
            # cos_trained_latent_col = torch.zeros(nclass, nclass, dtype=torch.float)
            proto_clone = proto.clone()
            set_trained_boundaries = []
            batch_l2_trained_diff = []
            set_latent_boundaries = []
            with torch.no_grad():
                normed_protos = transformDict['norm'](proto_clone.clone())
                protos_latent, protos_logits = model(normed_protos)
            for i in range(nclass):
                trained_boundaries = []
                latents_boundaries = []
                target_proto = torch.tensor([i], device= device)
                l2_trained_diff = []
                for k in range(nclass):
                    if i == k:
                        trained_boundaries.append((torch.zeros([3, 32, 32], device=device)))
                        latents_boundaries.append(torch.zeros(512, device=device))
                    if i!=k:
                        epoch = 1
                        start_proto = torch.unsqueeze(proto_clone[j], dim = 0)
                        last_loss, preds, probs = train_image_no_data(args, model=model, device=device,
                                                                   epoch=epoch, par_images=start_proto,
                                                               targets=target_proto, transformDict=transformDict)
                        print(f"Preds after {k} goes to {i}: {preds}\n")
                        if preds != i:
                            mispredictions.append([set, k, i, preds.item()])
                        model.eval()
                        last_loss_save[i][k] = last_loss
                        with open('{}/BOUNDARY_PROBS_{}.txt'.format(model_dir, date_time), 'a') as f:
                            f.write(f"Going from {k} to {i}, probabilities of {probs} \n\n")
                        f.close()
                        with torch.no_grad():
                            norm_trained_boundary = transformDict['norm'](start_proto.clone())
                            boundary_latent, boundary_logits = model(norm_trained_boundary)
                        start_proto = torch.squeeze(start_proto, dim=0)
                        print(f"Boundary  shape: {start_proto.shape}")
                        boundary_latent = torch.squeeze(boundary_latent, dim=0)
                        cos_trained_latent[i][k] = cos_sim(boundary_latent, protos_latent[i])
                        cos_trained_latent_col[i][k] = cos_sim(boundary_latent, protos_latent[k])
                        trained_boundaries.append(start_proto)
                        l2_trained_diff.append(torch.mean(torch.linalg.norm((boundary_latent.clone() - protos_latent[i]), dim=0)))
                        latents_boundaries.append(boundary_latent)

                set_trained_boundaries.append(torch.stack(trained_boundaries, dim=0))
                set_latent_boundaries.append(torch.stack(latents_boundaries, dim=0))
                batch_l2_trained_diff.append(torch.stack(l2_trained_diff, dim=0))
            cos_trained_latent_matrices.append(cos_trained_latent.clone())
            cos_trained_latent_col_matrices.append(cos_trained_latent_col.clone())
            stacked_sets_trained_boundaries.append(torch.stack(set_trained_boundaries, dim=0))
            stacked_trained_l2.append(torch.stack(batch_l2_trained_diff, dim=0))

            stacked_sets_latent_boundaries.append(torch.stack(set_latent_boundaries, dim=0))
      # combined_boundary_images = torch.mean(torch.stack(stacked_sets_trained_boundaries,dim=0), dim=0)
       # trained_boundary_sets.append(torch.stack(stacked_sets_trained_boundaries, dim=0))
       # print(len(trained_boundary_sets))
       # combined_boundary_latent = torch.mean(torch.stack(stacked_sets_latent_boundaries, dim=0), dim=0)
      #  batch_trained_cs, batch_trained_std = torch.std_mean(torch.stack(cos_trained_latent_matrices, dim=0), dim=0)
      #  batch_trained_col_cs, batch_trained_col_std = torch.std_mean(torch.stack(cos_trained_latent_col_matrices, dim=0), dim=0)
       # batch_diff_std.append(batch_trained_std)
      #  batch_diff_col_std.append(batch_trained_col_std)
      #  batch_cum_trained_cs, batch_cum_trained_cs_std = torch.std_mean(batch_trained_cs, dim=0)
        batch_cum_trained_cs, batch_cum_trained_cs_std = torch.std_mean(cos_trained_latent.clone(), dim=0)
     #   batch_cum_trained_col_cs, batch_cum_trained_col_cs_std = torch.std_mean(batch_trained_col_cs, dim=1)
        batch_cum_trained_col_cs, batch_cum_trained_col_cs_std = torch.std_mean(cos_trained_latent_col.clone(),dim=1)
        cum_trained_cs_avg = 1 - torch.mean(batch_cum_trained_cs)
        cum_trained_col_cs = 1 - torch.mean(batch_cum_trained_col_cs)
        final_comb_trained_cs_diffs.append(cum_trained_cs_avg.item())
        final_comb_trained_cols_cs_diffs.append(cum_trained_col_cs.item())
        final_comb_trained_cs_std.append(torch.mean(batch_cum_trained_cs_std).item())
        final_comb_trained_col_cs_std.append(torch.mean(batch_cum_trained_col_cs_std).item())
        final_ind_trained_col_cs_diffs.append([round(1 - val.item(), 4) for val in batch_cum_trained_col_cs])
        final_ind_trained_cs_col_stds.append([round(val.item(), 4) for val in batch_cum_trained_col_cs_std])
        final_ind_trained_cs_diffs.append([round(1 - val.item(), 4) for val in batch_cum_trained_cs])
        final_ind_trained_cs_diffs_std.append([round(val.item(), 4) for val in batch_cum_trained_cs_std])


        batch_trained_l2 = torch.mean(torch.stack(stacked_trained_l2, dim=0), dim=0)
        batch_cum_trained_l2 = torch.mean(batch_trained_l2, dim=0)
        final_comb_trained_l2_diffs.append(torch.mean(batch_cum_trained_l2))
        final_ind_trained_l2_diffs.append(batch_cum_trained_l2)


        class_diffs = []



        for t in range(1):
            matrix = stacked_sets_trained_boundaries
            for i in range(len(stacked_sets_trained_boundaries)):
                basefound = False
                inter_class_diffs = []
                base = 0
                for j in range(len(matrix)):
                    if basefound == False and j != i:
                        base = matrix[i][j].clone()
                        basefound == True
                        based_index = j
                    elif j != i:
                        inter_diff = cos_sim(matrix[i][j], matrix[i][base])
                        inter_latent = cos_sim(stacked_sets_latent_boundaries[i][j], stacked_sets_latent_boundaries[i][base])
                        inter_class_diffs.append([i, based_index, j, 1 - round(inter_diff.item(),4), 1 - round(inter_latent.item(), 2) ])
                    else:
                        inter_class_diffs.append([0.0,0.0,0.0,0.0,0.0])
            class_diffs.append(torch.stack(inter_class_diffs, dim=0))





       # print(f"shape of combined_image and combined_latent: {combined_boundary_images.shape} /n {combined_boundary_latent.shape}")

        with open('{}/Trained_Stats{}.txt'.format(saved_boundaries_path, date_time), 'a') as f:
            f.write("\n")
            f.write(
                f"Split {j} \n  average L2s: {batch_cum_trained_l2}\n average CS_diff: {[1-val for val in batch_cum_trained_cs]}\n,\
                 Cumulative L2: {torch.mean(batch_cum_trained_l2.clone())}\n Cumulative CS_diff: {cum_trained_cs_avg}")
            f.write("\n")
        f.close()

    # print(f"length of comb alphas : {len(final_comb_alphas_avg)}")
    # print(f"length of l2s : {len(final_ind_l2_diffs)}")
    # print(f"length of cs's : {len(final_ind_cs_diffs)}")

    with open('{}/BOUNDARY_{}.txt'.format(model_dir, date_time), 'a') as f:
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
        f.write(f" Final Loss Matrix: {last_loss_save} \n \n \
                Class Diffs: [Class, Based Boundary, Comparator, Image CS Diff, Latent CS Diff] : {class_diffs} \n \n \
                Matrix of Row-Wise CS diffs: {-(torch.sub(cos_trained_latent, 1))} \n \
                Matrix of Column-Wise CS diffs: {-(torch.sub(cos_trained_latent_col, 1))} \n \
                   row wise CS diffs: {final_ind_trained_cs_diffs[0]} \n \n \
                   row wise CS stds: {final_ind_trained_cs_diffs_std[0]} \n   \
                   column-wise CS diffs: {final_ind_trained_col_cs_diffs[0]} \n  \
                    column-wise CS stds: {final_ind_trained_cs_col_stds[0]} \n  \
                    \n \n cumulative row-wise CS diff: {final_comb_trained_cs_diffs[0]} \n  \
                     cumulative row-wise CS Std; {final_comb_trained_cs_std} \
                      \n  cumulative column-wise CS diff {final_comb_trained_cols_cs_diffs[0]} \
                       \n \n cumulative column-wise CS Std: {final_comb_trained_col_cs_std[0]} \n \n \
                        Mispredictions: {mispredictions}")

    f.close()


if __name__ == '__main__':
    main()
