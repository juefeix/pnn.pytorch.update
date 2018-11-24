import torch
import random
from dataloader import Dataloader
import utils
import os
from datetime import datetime
import argparse
import math
import numpy as np
from torch import nn
import models
import torch.optim as optim

result_path = "results/"
result_path = os.path.join(result_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))

parser = argparse.ArgumentParser(description='Your project title goes here')

# ======================== Data Setings ============================================
parser.add_argument('--dataset-test', type=str, default='CIFAR10', metavar='', help='name of training dataset')
parser.add_argument('--dataset-train', type=str, default='CIFAR10', metavar='', help='name of training dataset')
parser.add_argument('--dataroot', type=str, default='./data', metavar='', help='path to the data')
parser.add_argument('--save', type=str, default=result_path +'Save', metavar='', help='save the trained models here')
parser.add_argument('--logs', type=str, default=result_path +'Logs', metavar='', help='save the training log files here')
parser.add_argument('--resume', type=str, default=None, metavar='', help='full path of models to resume training')

# ======================== Network Model Setings ===================================

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--use_act', dest='use_act', action='store_true')
feature_parser.add_argument('--no-use_act', dest='use_act', action='store_false')
parser.set_defaults(use_act=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--unique_masks', dest='unique_masks', action='store_true')
feature_parser.add_argument('--no-unique_masks', dest='unique_masks', action='store_false')
parser.set_defaults(unique_masks=True)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--debug', dest='debug', action='store_true')
feature_parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.set_defaults(debug=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--train_masks', dest='train_masks', action='store_true')
feature_parser.add_argument('--no-train_masks', dest='train_masks', action='store_false')
parser.set_defaults(train_masks=False)

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--mix_maps', dest='mix_maps', action='store_true')
feature_parser.add_argument('--no-mix_maps', dest='mix_maps', action='store_false')
parser.set_defaults(mix_maps=False)

parser.add_argument('--filter_size', type=int, default=0, metavar='', help='use conv layer with this kernel size in FirstLayer')
parser.add_argument('--first_filter_size', type=int, default=0, metavar='', help='use conv layer with this kernel size in FirstLayer')
parser.add_argument('--nfilters', type=int, default=64, metavar='', help='number of filters in each layer')
parser.add_argument('--nmasks', type=int, default=1, metavar='', help='number of noise masks per input channel (fan out)')
parser.add_argument('--level', type=float, default=0.5, metavar='', help='noise level for uniform noise')
parser.add_argument('--scale_noise', type=float, default=1.0, metavar='', help='noise level for uniform noise')
parser.add_argument('--noise_type', type=str, default='uniform', metavar='', help='type of noise')
parser.add_argument('--dropout', type=float, default=0.5, metavar='', help='dropout parameter')
parser.add_argument('--net-type', type=str, default='resnet18', metavar='', help='type of network')
parser.add_argument('--act', type=str, default='relu', metavar='', help='activation function (for both perturb and conv layers)')
parser.add_argument('--pool_type', type=str, default='max', metavar='', help='pooling function (max or avg)')

# ======================== Training Settings =======================================
parser.add_argument('--batch-size', type=int, default=64, metavar='', help='batch size for training')
parser.add_argument('--nepochs', type=int, default=150, metavar='', help='number of epochs to train')
parser.add_argument('--nthreads', type=int, default=4, metavar='', help='number of threads for data loading')
parser.add_argument('--manual-seed', type=int, default=1, metavar='', help='manual seed for randomness')

# ======================== Hyperparameter Setings ==================================
parser.add_argument('--optim-method', type=str, default='SGD', metavar='', help='the optimization routine ')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='', help='learning rate')
parser.add_argument('--learning-rate-decay', type=float, default=None, metavar='', help='learning rate decay')
parser.add_argument('--momentum', type=float, default=0.9, metavar='', help='momentum')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='', help='weight decay')
parser.add_argument('--adam-beta1', type=float, default=0.9, metavar='', help='Beta 1 parameter for Adam')
parser.add_argument('--adam-beta2', type=float, default=0.999, metavar='', help='Beta 2 parameter for Adam')

args = parser.parse_args()
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
utils.saveargs(args)

class Model:
    def __init__(self, args):
        self.cuda = torch.cuda.is_available()
        self.lr = args.learning_rate
        self.dataset_train_name = args.dataset_train
        self.nfilters = args.nfilters
        self.batch_size = args.batch_size
        self.level = args.level
        self.net_type = args.net_type
        self.nmasks = args.nmasks
        self.unique_masks = args.unique_masks
        self.filter_size = args.filter_size
        self.first_filter_size = args.first_filter_size
        self.scale_noise = args.scale_noise
        self.noise_type = args.noise_type
        self.act = args.act
        self.use_act = args.use_act
        self.dropout = args.dropout
        self.train_masks = args.train_masks
        self.debug = args.debug
        self.pool_type = args.pool_type
        self.mix_maps = args.mix_maps

        if self.dataset_train_name.startswith("CIFAR"):
            self.input_size = 32
            self.nclasses = 10
            if self.filter_size < 7:
                self.avgpool = 4
            elif self.filter_size == 7:
                self.avgpool = 1

        elif self.dataset_train_name.startswith("MNIST"):
            self.nclasses = 10
            self.input_size = 28
            if self.filter_size < 7:
                self.avgpool = 14  #TODO
            elif self.filter_size == 7:
                self.avgpool = 7

        self.model = getattr(models, self.net_type)(
            nfilters=self.nfilters,
            avgpool=self.avgpool,
            nclasses=self.nclasses,
            nmasks=self.nmasks,
            unique_masks=self.unique_masks,
            level=self.level,
            filter_size=self.filter_size,
            first_filter_size=self.first_filter_size,
            act=self.act,
            scale_noise=self.scale_noise,
            noise_type=self.noise_type,
            use_act=self.use_act,
            dropout=self.dropout,
            train_masks=self.train_masks,
            pool_type=self.pool_type,
            debug=self.debug,
            input_size=self.input_size,
            mix_maps=self.mix_maps
        )

        self.loss_fn = nn.CrossEntropyLoss()

        if self.cuda:
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        if args.optim_method == 'Adam':
            self.optimizer = optim.Adam(parameters, lr=self.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)  #increase weight decay for no-noise large models
        elif args.optim_method == 'RMSprop':
            self.optimizer = optim.RMSprop(parameters, lr=self.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optim_method == 'SGD':
            self.optimizer = optim.SGD(parameters, lr=self.lr,  momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
            """
            # use this to set different learning rates for training noise masks and regular parameters:
            self.optimizer = optim.SGD([{'params': [param for name, param in self.model.named_parameters() if 'noise' not in name]},
                                        {'params': [param for name, param in self.model.named_parameters() if 'noise' in name], 'lr': self.lr * 10},
                                        ], lr=self.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True) #"""
        else:
            raise(Exception("Unknown Optimization Method"))


    def learning_rate(self, epoch):
        if self.dataset_train_name == 'CIFAR10':
            new_lr = self.lr * ((0.2 ** int(epoch >= 150)) * (0.2 ** int(epoch >= 250)) * (0.2 ** int(epoch >= 300)) * (0.2 ** int(epoch >= 350)) * (0.2 ** int(epoch >= 400))) # (1) Felix modified this
            # new_lr = self.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 90)) * (0.2 ** int(epoch >= 120)) * (0.2 ** int(epoch >= 160)))
        elif self.dataset_train_name == 'CIFAR100':
            new_lr = self.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120))* (0.1 ** int(epoch >= 160)))
        elif self.dataset_train_name == 'MNIST':
            new_lr = self.lr * ((0.2 ** int(epoch >= 30)) * (0.2 ** int(epoch >= 60))* (0.2 ** int(epoch >= 90)))
        elif self.dataset_train_name == 'FRGC':
            new_lr = self.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120))* (0.1 ** int(epoch >= 160)))
        elif self.dataset_train_name == 'ImageNet':
            decay = math.floor((epoch - 1) / 30)
            new_lr = self.lr * math.pow(0.1, decay)
            #print('\nReducing learning rate to {}\n'.format(new_lr))
        return new_lr


    def train(self, epoch, dataloader):
        self.model.train()

        lr = self.learning_rate(epoch+1)

        for param_group in self.optimizer.param_groups:
            #print(param_group)  #TODO figure out how to set diff learning rate to noise params if train_masks
            param_group['lr'] = lr

        losses = []
        accuracies = []
        for i, (input, label) in enumerate(dataloader):
            if self.cuda:
                label = label.cuda()
                input = input.cuda()

            output = self.model(input)
            loss = self.loss_fn(output, label)
            if self.debug:
                print('\nBatch:', i)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = output.data.max(1)[1]

            acc = pred.eq(label.data).cpu().sum()*100.0 / self.batch_size

            losses.append(loss.item())
            accuracies.append(acc)

        return np.mean(losses), np.mean(accuracies)

    def test(self, dataloader):
        self.model.eval()
        losses = []
        accuracies = []
        with torch.no_grad():
            for i, (input, label) in enumerate(dataloader):
                if self.cuda:
                    label = label.cuda()
                    input = input.cuda()

                output = self.model(input)
                loss = self.loss_fn(output, label)

                pred = output.data.max(1)[1]
                acc = pred.eq(label.data).cpu().sum()*100.0 / self.batch_size
                losses.append(loss.item())
                accuracies.append(acc)

        return np.mean(losses), np.mean(accuracies)

print('\n\n****** Creating {} model ******\n\n'.format(args.net_type))
setup = Model(args)
print('\n\n****** Preparing {} dataset *******\n\n'.format(args.dataset_train))
dataloader = Dataloader(args, setup.input_size)
loader_train, loader_test = dataloader.create()

# initialize model:
if args.resume is None:
    model = setup.model
    model.apply(utils.weights_init)
    train = setup.train
    test = setup.test
    init_epoch = 0
    acc_best = 0
    best_epoch = 0
    if os.path.isdir(args.save) == False:
        os.makedirs(args.save)
else:
    print('\n\nLoading model from saved checkpoint at {}\n\n'.format(args.resume))
    #self.model.load_state_dict(checkpoints.load(checkpoints.latest('resume')))
    setup.model = torch.load(args.resume)
    model = setup.model
    train = setup.train
    test = setup.test
    te_loss, te_acc = test(loader_test)
    init_epoch = int(args.resume.split('_')[3])  # extract N from 'results/xxx_xxx/Save/model_epoch_N_acc_nn.nn.pth'
    print('\n\nRestored Model Accuracy (epoch {:d}): {:.2f}\n\n'.format(init_epoch, te_acc))
    acc_best = te_acc
    best_epoch = init_epoch
    args.save = '/'.join(args.resume.split('/')[:-1])
    init_epoch += 1


print('\n\n****** Model Graph ******\n\n')
for arg in vars(model):
    print(arg, getattr(model, arg))

print('\n\nModel parameters:\n')
model_total = 0
for name, param in model.named_parameters():
    size = param.numel() / 1000000.
    print('{}  {}  requires_grad: {}  size: {:.2f}M'.format(name, list(param.size()), param.requires_grad, param.numel()/1000000.))
    model_total += size

print('\n\nNoise masks:\n')
masks_total = 0
for name, param in [(name, param) for name, param in model.named_parameters() if 'noise' in name]:
    size = param.numel() / 1000000.
    print('{:>22}  size: {:.2f}M'.format(str(list(param.size())), param.numel()/1000000.))
    masks_total += size

print('\n\nModel size: {:.2f}M regular parameters, {:.2f}M noise mask values\n\n'.format(model_total - masks_total, masks_total))
"""
print('\n\n******************** Model parameters:\n')
for param in model.parameters():
    #if param.requires_grad:
    print('{} {}'.format(list(param.size()), param.requires_grad))
    
print('\n\n****** Model state_dict() ******\n\n')
for name, param in model.state_dict().items():
    print('{}  {}  {}'.format(name, list(param.size()), param.requires_grad))
"""

print('\n\n****** Model Configuration ******\n\n')
for arg in vars(args):
    print(arg, getattr(args, arg))

if args.net_type != 'resnet18' and args.net_type != 'noiseresnet18' and (args.first_filter_size == 0 or args.filter_size == 0):
    if args.train_masks:
        msg = '(also training noise masks values)'
    else:
        msg = '(noise masks are fixed)'
else:
    msg = ''

print('\n\nTraining {} model {}\n\n'.format(args.net_type, msg))

accuracies = []

for epoch in range(init_epoch, args.nepochs, 1):

    tr_loss, tr_acc = train(epoch, loader_train)
    te_loss, te_acc = test(loader_test)

    accuracies.append(te_acc)

    if te_acc > acc_best and epoch > 10:
        print('{}  Epoch {:d}/{:d}  Train: Loss {:.2f} Accuracy {:.2f} Test: Loss {:.2f} Accuracy {:.2f} (best result, saving to {})'.format(
                        str(datetime.now())[:-7], epoch, args.nepochs, tr_loss, tr_acc, te_loss, te_acc, args.save))
        model_best = True
        acc_best = te_acc
        best_epoch = epoch
        torch.save(model, args.save + '/model_epoch_{:d}_acc_{:.2f}.pth'.format(epoch, te_acc))
    else:
        if epoch == 0:
            print('\n')
        print('{}  Epoch {:d}/{:d}  Train: Loss {:.2f} Accuracy {:.2f} Test: Loss {:.2f} Accuracy {:.2f}'.format(
                                str(datetime.now())[:-7], epoch, args.nepochs, tr_loss, tr_acc, te_loss, te_acc))

print('\n\nBest Accuracy: {:.2f}  (epoch {:d})\n\n'.format(acc_best, best_epoch))

print('\n\nTest Accuracies:\n\n')

for v in accuracies:
    print('{:.2f}'.format(v)+', ', end='')
print('\n\n')

plot = False
if plot:
    import matplotlib.pyplot as plt
    plt.plot(range(args.nepochs), accuracies, 'black', label='model_1')
    plt.plot(range(args.nepochs), accuracies, 'red', label='model_2')
    plt.plot(range(args.nepochs), accuracies, 'blue', label='model_3')
    plt.title('Test Accuracy (CIFAR-10)', fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('%', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='center right', prop={'size': 14})
    plt.show()