# dataloader.py

import os
import torch
import datasets
#import torch.utils.data
import torchvision.transforms as transforms

class Dataloader:

    def __init__(self, args, input_size):
        self.args = args

        self.dataset_test_name = args.dataset_test
        self.dataset_train_name = args.dataset_train
        self.input_size = input_size

        if self.dataset_train_name == 'LSUN':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(db_path=args.dataroot, classes=['bedroom_train'],
                transform=transforms.Compose([
                    transforms.Scale(self.input_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_train_name == 'CIFAR10' or self.dataset_train_name == 'CIFAR100':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(root=self.args.dataroot, train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(self.input_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
                )

        elif self.dataset_train_name == 'CocoCaption' or self.dataset_train_name == 'CocoDetection':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(root=self.args.dataroot, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Scale(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_train_name == 'STL10' or self.dataset_train_name == 'SVHN':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(root=self.args.dataroot, split='train', download=True,
                transform=transforms.Compose([
                    transforms.Scale(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_train_name == 'MNIST':
            self.dataset_train = getattr(datasets, self.dataset_train_name)(root=self.args.dataroot, train=True, download=True,
                transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
                )

        elif self.dataset_train_name == 'ImageNet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            self.dataset_train = datasets.ImageFolder(root=os.path.join(self.args.dataroot,self.args.input_filename_train),
                transform=transforms.Compose([
                    transforms.RandomSizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                   ])
                )

        elif self.dataset_train_name == 'FRGC':
            self.dataset_train = datasets.ImageFolder(root=self.args.dataroot+self.args.input_filename_train,
                transform=transforms.Compose([
                    transforms.Scale(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])
                )

        elif self.dataset_train_name == 'Folder':
            self.dataset_train = datasets.ImageFolder(root=self.args.dataroot+self.args.input_filename_train,
                transform=transforms.Compose([
                    transforms.Scale(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_train_name == 'FileList':
            self.dataset_train = datasets.FileList(self.input_filename_train, self.label_filename_train, self.split_train,
                self.split_test, train=True,
                transform_train=transforms.Compose([
                    transforms.Scale(self.input_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                transform_test=transforms.Compose([
                    transforms.Scale(self.input_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                loader_input=self.loader_input,
                loader_label=self.loader_label,
                )

        elif self.dataset_train_name == 'FolderList':
            self.dataset_train = datasets.FileList(self.input_filename_train, self.label_filename_train, self.split_train,
                self.split_test, train=True,
                transform_train=transforms.Compose([
                    transforms.Scale(self.input_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                transform_test=transforms.Compose([
                    transforms.Scale(self.input_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                loader_input=self.loader_input,
                loader_label=self.loader_label,
                )

        else:
            raise(Exception("Unknown Dataset"))

        if self.dataset_test_name == 'LSUN':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(db_path=args.dataroot, classes=['bedroom_val'],
                transform=transforms.Compose([
                    transforms.Scale(self.input_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )
        
        elif self.dataset_test_name == 'CIFAR10' or self.dataset_test_name == 'CIFAR100':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(root=self.args.dataroot, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
                )

        elif self.dataset_test_name == 'CocoCaption' or self.dataset_test_name == 'CocoDetection':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(root=self.args.dataroot, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Scale(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_test_name == 'STL10' or self.dataset_test_name == 'SVHN':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(root=self.args.dataroot, split='test', download=True,
                transform=transforms.Compose([
                    transforms.Scale(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_test_name == 'MNIST':
            self.dataset_test = getattr(datasets, self.dataset_test_name)(root=self.args.dataroot, train=False, download=True, 
                transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])
                )

        elif self.dataset_test_name == 'ImageNet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            self.dataset_test = datasets.ImageFolder(root=os.path.join(self.args.dataroot,self.args.input_filename_test),
                transform=transforms.Compose([
                    transforms.Scale(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                   ])
                )

        elif self.dataset_test_name == 'FRGC':
            self.dataset_test = datasets.ImageFolder(root=self.args.dataroot+self.args.input_filename_test,
                transform=transforms.Compose([
                    transforms.Scale(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])
                )

        elif self.dataset_test_name == 'Folder':
            self.dataset_test = datasets.ImageFolder(root=self.args.dataroot+self.args.input_filename_test,
                transform=transforms.Compose([
                    transforms.Scale(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ])
                )

        elif self.dataset_test_name == 'FileList':
            self.dataset_test = datasets.FileList(self.input_filename_test, self.label_filename_test, self.split_train,
                self.split_test, train=True,
                transform_train=transforms.Compose([
                    transforms.Scale(self.input_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                loader_input=self.loader_input,
                loader_label=self.loader_label,
                )

        elif self.dataset_test_name == 'FolderList':
            self.dataset_test = datasets.FileList(self.input_filename_test, self.label_filename_test, self.split_train,
                self.split_test, train=True,
                transform_train=transforms.Compose([
                    transforms.Scale(self.input_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
                loader_input=self.loader_input,
                loader_label=self.loader_label,
                )
            
        else:
            raise(Exception("Unknown Dataset"))

    def create(self, flag=None):
        if flag == "Train":
            dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.args.batch_size,
                shuffle=True, num_workers=int(self.args.nthreads), pin_memory=True)
            return dataloader_train

        if flag == "Test":
            dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.args.batch_size,
                shuffle=False, num_workers=int(self.args.nthreads), pin_memory=True)
            return dataloader_test

        if flag == None:
            dataloader_train = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.args.batch_size,
                shuffle=True, num_workers=int(self.args.nthreads), pin_memory=True)
        
            dataloader_test = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.args.batch_size,
                shuffle=False, num_workers=int(self.args.nthreads), pin_memory=True)
            return dataloader_train, dataloader_test