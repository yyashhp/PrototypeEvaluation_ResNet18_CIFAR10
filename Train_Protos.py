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

    #MEAN = [0.4914, 0.4822, 0.4465]
    MEAN = [0.5] * 3
    STD = [0.5] * 3
    H, W = 32, 32
    #STD = [0.2471, 0.2435, 0.2616]
    data_schedule = [0.25, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    transformDict = {}
    transformDict['norm'] = transforms.Compose([transforms.Normalize(MEAN, STD)])
    transformDict['basic'] = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(H, padding=4),transforms.Normalize(MEAN, STD)])

    nclass = 100
    nchannels = 3


    par_targets = torch.arange(nclass, dtype=torch.long, device=device)
    par_image_tensors = []
    saved_protos = []
    L2_cum_latent_means = []
    L2_cum_image_means = []
    CS_adv_image = []
    CS_adv_latent = []
    CS_means = []
    CS_mean_no_zero = []
    for run in range(args.total_runs):
        with torch.no_grad():
            par_image_fresh = torch.rand([nclass,nchannels,H, W ], dtype=torch.float, device=device)
            par_image_tensors.append(par_image_fresh.clone())
    for j in range(len(data_schedule)):
        model = ResNet18(nclass=nclass, scale=args.model_scale, channels=nchannels, **kwargsUser).to(device)
        model_saved = torch.load(f"{saved_model_path}/{j}_Saved_Model_with_{data_schedule[j]}_CIFAR100_Data_0621_13_24_49", map_location=device)
        model.load_state_dict(model_saved)
        model.multi_out = 1
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        for run in range(args.total_runs):
            for epoch in range(1):
                last_loss, preds, probs = train_image_no_data(args, model=model,
                                                              device=device,
                                                              epoch = epoch,
                                                              par_images=par_image_tensors[run],
                                                              targets = par_targets,
                                                              transformDict=transformDict)
                if epoch == 0:
                    print(last_loss)
        torch.save(par_image_tensors, f"{saved_protos_path}/CIFAR_100_Final_Saved_Protos_SPLIT_{j}")
        saved_protos.append(par_image_tensors)
        # cos similarites
        cos_matrices = []
        for proto in par_image_tensors:
            par_tensors_norm = transformDict['norm'](proto.clone())
            latent_p, logits_p = model(par_tensors_norm)

            # compute cos similarity matrix

            cos_mat_latent_temp = torch.zeros(nclass, nclass, dtype=torch.float)
            cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)

            for i in range(len(latent_p)):
                for q in range(len(latent_p)):
                    if i != q:
                        cos_mat_latent_temp[i, q] = cos_sim(latent_p[i].view(-1), latent_p[q].view(-1))
            cos_matrices.append(cos_mat_latent_temp.clone())

        cos_mat_std, cos_mat_mean = torch.std_mean(torch.stack(cos_matrices, dim=0), dim=0)
        CS_means.append(1-(torch.mean(cos_mat_mean.clone())))
        CS_mean_no_zero.append(1-((torch.sum(cos_mat_mean.clone())) / ((nclass * nclass) - nclass)))
        with open('{}/CIFAR_100_LOADED_CS_stats_{}.txt'.format(model_dir, date_time), 'a') as f:
            f.write("\n")
            f.write(
                f"Training split: {j}, \t Each Protos CS_Similarity_Mean, {cos_mat_mean.clone()} \t Overall CS_Diff_Mean: {1-torch.mean(cos_mat_mean.clone())}\t CS_diff_mean with no zeroes: {1-((torch.sum(cos_mat_mean.clone())) / ((nclass * nclass) - nclass))}\n")
            f.write("\n")
        f.close()

        "'Write means to file...****"

        # l2 similarity with adversarial images...

        L2_image_means = []
        L2_latent_means = []
        CS_image_means = []
        CS_latent_means = []

        for proto in par_image_tensors:
            proto_copy = proto.clone()
            with torch.no_grad():
                proto_copy_norm = transformDict['norm'](proto_copy)
                latent_onehot, logit_onehot = model(proto_copy_norm)
            model.eval()
            model.multi_out = 0
            attack = L2DeepFoolAttack(overshoot=0.02)
            preprocessing = dict(mean=MEAN, std=STD, axis=-3)
            fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

            print('Computing DF L2_stats')

            raw, new_proto, is_adv = attack(fmodel, proto_copy.clone(),
                                            torch.arange(nclass, dtype=torch.long, device=device), epsilons=10.0)

            model.multi_out = 1

            with torch.no_grad():
                new_proto_norm = transformDict['norm'](new_proto.clone())
                latent_adv, logits_adv = model(new_proto_norm)
            L2_df_image = torch.linalg.norm((new_proto - proto_copy).view(nclass, -1), dim=1)
            L2_df_latent = torch.linalg.norm((latent_adv - latent_onehot).view(nclass, -1), dim=1)
            CS_df_image = F.cosine_similarity(new_proto.view(nclass, -1), proto_copy.view(nclass, -1))
            CS_df_latent = F.cosine_similarity(latent_adv.view(nclass, -1), latent_onehot.view(nclass, -1))

            im_df_std, im_df_mean = torch.std_mean(L2_df_image)
            latent_df_std, latent_df_mean = torch.std_mean(L2_df_latent)
            L2_image_means.append(im_df_mean.clone())
            L2_latent_means.append(latent_df_mean.clone())
            CS_image_means.append(torch.mean(CS_df_image).clone())
            CS_latent_means.append(torch.mean(CS_df_latent).clone())
            with open('{}/CIFAR_100_LOADED_Adv_stats_{}.txt'.format(model_dir, date_time), 'a') as f:
                f.write("\n")
                f.write(
                    "Training split: {}, \t L2 image and latent means: {} \t {} \t CS image and latent means: {} \t {}  ".format(
                        j, im_df_mean.clone(), latent_df_mean.clone(), torch.mean(CS_df_image).clone(),
                        torch.mean(CS_df_latent).clone()))
                f.write("\n")
            f.close()

        L2_cum_image_std, L2_cum_image_mean = torch.std_mean(torch.stack(L2_image_means, dim=0), dim=0)
        L2_cum_image_means.append(L2_cum_image_mean.clone())

        L2_cum_latent_std, L2_cum_latent_mean = torch.std_mean(torch.stack(L2_latent_means, dim=0), dim=0)
        L2_cum_latent_means.append(L2_cum_latent_mean.clone())

        CS_df_std, CS_df_mean = torch.std_mean(torch.stack(CS_image_means, dim=0), dim=0)
        CS_adv_image.append(CS_df_mean.clone())
        CS_latent_std, CS_latent_mean = torch.std_mean(torch.stack(CS_latent_means, dim=0), dim=0)
        CS_adv_latent.append(1-(CS_latent_mean.clone()))
        with open('{}/CIFAR_100_LOADED_Adv_stats_{}.txt'.format(model_dir, date_time), 'a') as f:
            f.write("\n")
            f.write(f"Split {j} \t L2_diff latent overall mean: {L2_cum_latent_mean}\t \
                    CS_diff latent overall mean: {1-CS_latent_mean}\n")
        f.close()

    with open('{}/CIFAR_100_LOADED_final_data_summary_{}.txt'.format(model_dir, date_time), 'a') as f:
        f.write("Data  \t CS_norm metric \t L2 adversarial latent means \
                \t L2 adversarial image means \t CS adversarial image means\
                 \t CS adversarial latent means  \n")
        for i in range(len(data_schedule)):
            f.write("{0:4.4f} \t {1:4.4f}\t {2:4.4f}\t {3:4.4f}\t {4:4.4f}\t {5:4.4f} \t \n".format(data_schedule[i], CS_means[i]
                                        ,L2_cum_latent_means[i], L2_cum_image_means[i]
                                         , CS_adv_image[i], CS_adv_latent[i]))
    f.close()


if __name__ == '__main__':
    main()
