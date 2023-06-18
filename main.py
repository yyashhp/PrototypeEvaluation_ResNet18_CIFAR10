from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms, models

import torchvision.transforms as transforms
import json
import datetime
from datetime import datetime
import numpy as np

import os
import argparse
import ResNet18Model
from ResNet18Model import Block
from ResNet18Model import ResNet
from ResNet18Model import ResNet18
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import L2DeepFoolAttack

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

#runs default 5


args = parser.parse_args()

kwargsUser = {}
kwargsUser['norm_type'] = args.norm_type
network_string = "ResNet18"

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

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(full_dir_plot):
    os.makedirs(full_dir_plot)

with open('{}/commandline_args.txt'.format(model_dir), 'a') as f:
    json.dump(args.__dict__, f, indent=2)
f.close()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

torch.manual_seed(args.seed)

kwargs = {'num_workers': 4, 'pin_memory': False} if use_cuda else {}
torch.cuda.empty_cache()

print ("cuda: ", use_cuda)


def eval_train(model, device, train_loader, transformDict):
    model.eval()
    train_loss = 0
    correct = 0
    curr = model.multi_out
    model.multi_out = 0
    transformDictionary = transformDict
    with torch.inference_mode():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device) 
            data = transformDictionary['norm'](data)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss /= len(train_loader.dataset)
        print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
        training_accuracy = correct / len(train_loader.dataset)
        model.multi_out = curr
        return train_loss, training_accuracy


def train_image_no_data(args, model, device, epoch, par_images, targets, transformDict):
    print ("training images against one hot encodings")
    for batch_idx in range(20):
        _par_images_opt = par_images.clone().detach().requires_grad_(True).to(device)

        _par_images_opt_norm = transformDict['norm'](_par_images_opt)

        L2_img, logits_img = model(_par_images_opt_norm)

        loss = F.cross_entropy(logits_img, targets, reduction='none')
        print(f"gradient used: {torch.ones_like(loss)}")
        loss.backward(gradient=torch.ones_like(loss))

        with torch.no_grad():
            gradients_unscaled = _par_images_opt_norm.grad.clone()
            grad_mag = gradients_unscaled.view(gradients_unscaled.shape[0], -1).norm(2, dim=-1)
            print(f"Grad_Mag:{grad_mag}")
            image_grads = 0.01 * gradients_unscaled / grad_mag.view(-1, 1, 1, 1)
           # image_gradients = torch.nan_to_num(image_grads)
            #print(f"Printing image gradients here: {image_gradients}")
            if torch.mean(loss) > 1e-7:
                par_images.add_(-image_grads)
            par_images.clamp_(0.0, 1.0)
            print(f"Printing par_image_vals: {par_images}")

            _par_images_opt.grad.zero_()
    if batch_idx % args.log_interval == 0:
        print('Train Epoch: {}\t BatchID: {}\t Loss {:.6f}'.format(epoch, batch_idx, torch.mean(loss).item()))

    with torch.no_grad():
        _par_images_final = par_images.clone().detach().requires_grad_(False).to(device)
        _par_images_final_norm = transformDict['norm'](_par_images_final)
        L2_img, logits_img = model(_par_images_final_norm)
        pred = logits_img.max(1, keepdim=True)[1]
        probs = F.softmax(logits_img)

    return loss, pred, probs


def train(model, device, optimizer, criterion, cur_loader, epoch, max_steps, scheduler, transformDict):

    model.train()
    print('Training model')
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(cur_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs_norm = transformDict['norm'](inputs)
        p, outputs = model(inputs_norm)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx < max_steps:
            scheduler.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(cur_loader.dataset),
                       100. * batch_idx / len(cur_loader), loss.item()))

def eval_test(model, device, test_loader, transformDict):
    model.eval()
    test_loss = 0
    correct = 0
    curr = model.multi_out
    model.multi_out = 0
    transformDictionary = transformDict
    with torch.inference_mode():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = transformDictionary['norm'](data)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    model.multi_out = curr
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    #decrease learning rate
    lr = args.lr
    if epoch >= (0.5 * args.epochs):
        lr = args.lr * 0.1
    if epoch >= (0.75 * args.epochs):
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def main():
    #Setting up dataloader
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2471, 0.2435, 0.2616]

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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=gen_transform_test)


    
    kwargsUser['num_classes'] = 10
    nclass = 10
    nchannels = 3
    H, W = 32, 32

    splits = []

    all_inds = np.arange(len(trainset.targets))

    inds_train1, inds_test1, y_train1, y_test1 = train_test_split(all_inds, trainset.targets, test_size=0.25,
                                                                  random_state=args.seed, stratify=trainset.targets)

    splits.append(inds_test1)

    inds_train2, inds_test2, y_train2, y_test2 = train_test_split(all_inds, trainset.targets, test_size=0.4,
                                                                  random_state=args.seed, stratify=trainset.targets)

    splits.append(inds_test2)

    inds_train3, inds_test3, y_train3, y_test3 = train_test_split(all_inds, trainset.targets, test_size=0.6,
                                                                  random_state=args.seed, stratify=trainset.targets)

    splits.append(inds_test3)

    inds_train4, inds_test4, y_train4, y_test4 = train_test_split(all_inds, trainset.targets, test_size=0.7,
                                                                  random_state=args.seed, stratify=trainset.targets)

    splits.append(inds_test4)

    inds_train5, inds_test5, y_train5, y_test5 = train_test_split(all_inds, trainset.targets, test_size=0.8,
                                                                  random_state=args.seed, stratify=trainset.targets)

    splits.append(inds_test5)

    inds_train6, inds_test6, y_train6, y_test6 = train_test_split(all_inds, trainset.targets, test_size=0.9,
                                                                  random_state=args.seed, stratify=trainset.targets)

    splits.append(inds_test6)

    # add 100% training
    splits.append(all_inds)

    p_magnitude = 0.03

    CS_means = []
    L2_cum_image_means = []
    L2_cum_latent_means = []
    CS_adv_image = []
    CS_adv_latent = []
    saved_protos = []


    test_accs = []
    par_image_tensors = []
    par_targets = torch.arange(nclass, dtype=torch.long, device=device)



    for run in range(args.total_runs):
        with torch.no_grad():
            par_images_metric = torch.rand([kwargsUser['num_classes'], nchannels, H, W], dtype=torch.float, device=device)
            par_image_tensors.append(par_images_metric.clone())
    model = ResNet18(nclass=nclass, scale=args.model_scale, channels=nchannels, **kwargsUser).to(device)
    for j in range(len(splits)):
        subtrain = torch.utils.data.Subset(trainset, splits[j])
        print(f"Length of subtrain: {len(subtrain)}")

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        model.eval()
        initial_loss_train, initial_acc_train = eval_train(model, device, train_loader, transformDict)
        initial_loss_test, initial_acc_test = eval_test(model, device, test_loader, transformDict)

        with open('{}/train_hist_{}.txt'.format(model_dir, date_time), 'a') as f:
            f.write("=====INITIAL======\n")
            f.write("{0:4.3f}\t{1:4.3f}\t{2:4.0f}\t{3:4.3f}\t{4:6.5f}\n".format(initial_acc_train,initial_acc_test,len(train_loader.dataset),initial_loss_train,initial_loss_test))
            f.write("==================\n")
        f.close()


        model.train()
        model.multi_out = 1
        for p in model.parameters():
            p.requires_grad = True

        lr_init = args.lr
        print(f"Learning rate: {lr_init}")

        optimizer = optim.SGD(model.parameters(), lr = lr_init, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

        steps_per_epoch = int(np.ceil(len(train_loader.dataset) / args.batch_size))

        print("len(cur_loader.dataset)", len(train_loader.dataset))
        print("len(cur_loader)", len(train_loader))

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train_loader), eta_min=0.0000001, last_epoch=-1, verbose=False)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1, args.epochs+1):
            model.train()
            model.multi_out = 1
            train(model=model, device=device, epoch=epoch, max_steps=steps_per_epoch, scheduler=scheduler, criterion=criterion, cur_loader=train_loader, optimizer=optimizer, transformDict=transformDict)

            model.multi_out = 0
    #Training model
            print('================================================================')
            loss_train, acc_train = eval_train(model, device, train_loader, transformDict)
            loss_test, acc_test = eval_test(model, device, test_loader, transformDict)

            with open('{}/train_hist_{}.txt'.format(model_dir, date_time), 'a') as f:
                f.write("{0:4.3f}\t{1:4.3f}\t{2:4.0f}\t{3:4.3f}\t{4:6.5f}\n".format(acc_train,acc_test,len(train_loader.dataset),loss_train,loss_test))
            f.close()
            if epoch == args.epochs:
                torch.save(model.state_dict(),os.path.join(model_dir, 'model-{}-epoch{}-training{}.pt'.format(network_string,epoch,j)))
        test_accs.append(acc_test)
#Saving trained model
        model.multi_out = 1
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
#Freezing model for protos
        for run in range(args.total_runs):
            for epoch in range(1, (args.epochs + 1)):
                last_loss, preds, probs = train_image_no_data(args, model = model, device = 'cuda', epoch = epoch, par_images=par_image_tensors[run], targets = par_targets, transformDict=transformDict)
        # Protos are trained here^

        saved_protos.append(par_image_tensors)
        #cos similarites
        cos_matrices = []
        for proto in par_image_tensors:
            par_tensors_norm = transformDict['norm'](proto.clone())
            latent_p, logits_p = model(par_tensors_norm)

            #compute cos similarity matrix

            cos_mat_latent_temp = torch.zeros(nclass, nclass, dtype=torch.float)
            cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)

            for i in range(len(latent_p)):
                for q in range(len(latent_p)):
                    if i != q:
                        cos_mat_latent_temp[i,q] = cos_sim(latent_p[i].view(-1),latent_p[q].view(-1))
                        print(cos_mat_latent_temp[i,q])
            cos_matrices.append(cos_mat_latent_temp.clone())

        cos_mat_std, cos_mat_mean = torch.std_mean(torch.stack(cos_matrices, dim=0), dim=0)
        CS_means.append(torch.mean(cos_mat_mean.clone()))
        with open('{}/CS_stats_{}.txt'.format(model_dir, date_time), 'a') as f:
            f.write("\n")
            f.write(
                "Training split: {}, \t CS_diff_mean {}  ".format(
                    j,torch.mean(cos_mat_mean.clone())))
            f.write("\n")
        f.close()

        "'Write means to file...****"

        #l2 similarity with adversarial images...

        L2_image_means = []
        L2_latent_means = []
        CS_image_means = []
        CS_latent_means = []


        for proto in par_image_tensors:
            proto_copy = proto.clone()
            print("Printing max of proto and its copy:")
            print(torch.max(proto))
            print(torch.max(proto_copy))
            print("Printing min of proto and then its copy")
            print(torch.min(proto))
            print(torch.min(proto_copy))
            with torch.no_grad():
                proto_copy_norm = transformDict['norm'](proto_copy)
                latent_onehot, logit_onehot = model(proto_copy_norm)
            model.eval()
            model.multi_out = 0
            attack = L2DeepFoolAttack(overshoot=0.02)
            preprocessing = dict(mean=MEAN, std=STD, axis=-3)
            fmodel = PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing)

            print('Computing DF L2_stats')

            raw, new_proto, is_adv = attack(fmodel, proto_copy.clone(), torch.arange(nclass, dtype=torch.long, device=device), epsilons=10.0)

            model.multi_out = 1

            with torch.no_grad():
                new_proto_norm = transformDict['norm'](new_proto.clone())
                latent_adv, logits_adv = model(new_proto_norm)
            L2_df_image = torch.linalg.norm((new_proto-proto_copy).view(nclass, -1), dim=1)
            L2_df_latent = torch.linalg.norm((latent_adv - latent_onehot).view(nclass, -1), dim=1)
            CS_df_image = F.cosine_similarity(new_proto.view(nclass, -1), proto_copy.view(nclass, -1))
            CS_df_latent = F.cosine_similarity(latent_adv.view(nclass, -1), latent_onehot.view(nclass, -1))

            im_df_std, im_df_mean = torch.std_mean(L2_df_image)
            latent_df_std, latent_df_mean = torch.std_mean(L2_df_latent)
            L2_image_means.append(im_df_mean.clone())
            L2_latent_means.append(latent_df_mean.clone())
            CS_image_means.append(torch.mean(CS_df_image).clone())
            CS_latent_means.append(torch.mean(CS_df_latent).clone())
            with open('{}/Adv_stats_{}.txt'.format(model_dir, date_time), 'a') as f:
                f.write("\n")
                f.write("Training split: {}, \t L2 image and latent means: {} \t {} \t CS image and latent means: {} \t {}  ".format(j,im_df_mean.clone(), latent_df_mean.clone(), torch.mean(CS_df_image).clone(), torch.mean(CS_df_latent).clone() ))
                f.write("\n")
            f.close()

        L2_cum_image_std, L2_cum_image_mean = torch.std_mean(torch.stack(L2_image_means, dim=0), dim=0)
        L2_cum_image_means.append(L2_cum_image_mean.clone())

        L2_cum_latent_std, L2_cum_latent_mean = torch.std_mean(torch.stack(L2_latent_means, dim=0), dim=0)
        L2_cum_latent_means.append(L2_cum_latent_mean.clone())

        CS_df_std, CS_df_mean = torch.std_mean(torch.stack(CS_image_means, dim=0), dim=0)
        CS_adv_image.append(CS_df_mean.clone())
        CS_latent_std, CS_latent_mean = torch.std_mean(torch.stack(CS_latent_means, dim=0), dim=0)
        CS_adv_latent.append(CS_latent_mean.clone())


   # data_schedule = [0.25, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]

#    with open('{}/final_data_summary_{}.txt'.format(model_dir, date_time), 'a') as f:
 #       f.write("Data \t Test Acc \t Prototype Tensors \t CS_norm metric \t L2 adversarial latent means \t L2 adversarial image means \t CS adversarial image means \t CS adversarial latent means  \n")
  #      for i in range(len(data_schedule)):
   #         f.write("{0:4.4f} \t {1:4.4f}\t {}\t {3:4.4f}\t {4:4.4f}\t {5:4.4f}\t {6:4.4f}\t {7:4.4f} \n".format(data_schedule[i], test_accs[i],saved_protos[i], CS_means[i],L2_cum_latent_means[i], L2_cum_image_means[i],CS_adv_image[i], CS_adv_latent[i]))
    #f.close()


    print("-----------------------------------------------")
    print("Now calculating boundary tensors")

    boundaries = []
    alphas_needed = torch.zeros(nclass, nclass, dtype = torch.int).to(device)
    model.eval()
    model.multi_out = 0
    anomalies = []
    for i in range(nclass):
        for j in range(nclass):
            if i == j:
                boundaries.append(0)
            elif i != j:
                tester_weights = []
                #preds = []

                starter = par_image_tensors[0][i]
                end = par_image_tensors[0][j]
                starter_copy = starter.clone().detach().requires_grad_(False).to(device)
                end_copy = end.clone().detach().requires_grad_(False).to(device)
                starter_norm = transformDict['norm'](starter_copy)
                end_norm = transformDict['norm'](end_copy)
                starter_resize = torch.unsqueeze(starter_copy, dim=0)
                end_resize = torch.unsqueeze(end_copy, dim=0)
                starter_norm_resize = torch.unsqueeze(starter_norm, dim=0)
                end_norm_resize = torch.unsqueeze(end_norm, dim=0)
#                print(starter_norm_resize.shape)
 #               print(end_norm_resize.shape)
                with torch.inference_mode():
                    starter_logits = model(starter_norm_resize)
                    end_logits = model(end_norm_resize)
                    start_pred = starter_logits.max(1, keepdim=True)[1]
                    end_pred = end_logits.max(1, keepdim=True)[1]
                    print(start_pred)
                    print(end_pred)
                    start_probs = F.softmax(starter_logits, dim=1)
                    end_probs = F.softmax(end_logits, dim=1)
                    print("Printing start and end probs")
                    print(start_probs)
                    print(end_probs)
                #preds.append(start_pred.clone())
                for alpha in range (1,20):
                    adj_alpha = 1/(alpha)
                    weighted_starter = torch.mul(starter_resize, (adj_alpha))
                    weighted_end = torch.mul(end_resize, (1-adj_alpha))
                    tester = torch.add(weighted_starter, weighted_end, alpha = 1)
                    print(tester.shape)
                    tester_norm = transformDict['norm'](tester)
                    tester_logits = model(tester_norm)
                #    curr_probs = F.softmax(tester_logits)
                    curr_pred = torch.argmax(tester_logits, dim = 1, keepdim=True)
                    tester_weights.append(tester_norm.clone())
                    if curr_pred == end_pred:
                        print(f"Alpha here is {alpha}, boundaries are {i}, {j}")
                        half_curr = torch.mul(tester, 0.5)
                        if alpha == 1:
                            prev_ind = 0
                        else:
                            prev_ind = alpha-2
                        half_prev = torch.mul(tester_weights[prev_ind], 0.5)
                        print("Printing summed half curr and half prev sizes")
                        median = torch.add(half_curr, half_prev, alpha = 1)
                        print(median.shape)
                        boundaries.append([i, j, median])
                        alphas_needed[i][j] = alpha
                        #preds.append(curr_pred)
                 #   elif curr_pred == start_pred:
                       # tester_weights.append(tester_norm.clone())
                        #preds.append(curr_pred)
                    elif (curr_pred != end_pred and curr_pred != start_pred):
                        anomalies.append([start_pred, end_pred, curr_pred, alpha])
    count = 0
    for i in range(nclass):
        primary = par_image_tensors[0][i]
        print(f"Proto tensor of class {i}, {primary.view(-1)}")
        vals, indices = primary.clone().sort(dim=0)
        inds = list(range(len(vals)))


        for j in range(nclass):
            if i == j:
                continue
            else:
   
                print(f"Alphas needed for boundary between {i} and {j} : {alphas_needed[i,j]}")
                print(f"tensor there: {boundaries[0]}")
            count += 1







    
    




if __name__ == '__main__':
    main()
