import torch
import utils
import numpy as np
from torch import nn
import torch.optim as optim
import datasets
import torchvision.transforms as transforms


class Net(nn.Module):
	def __init__(self, ):
		super(Net, self).__init__()
		self.linear = nn.Linear(9 * 6 * 6, 10)
		self.noise = nn.Parameter(torch.Tensor(1, 1, 28, 28), requires_grad=True)
		self.noise.data.uniform_(-1, 1)

		self.layers = nn.Sequential(
			nn.Conv2d(1, 9, kernel_size=5, stride=2, bias=False),
			nn.MaxPool2d(2, 2),
			nn.ReLU(),
		)

	def forward(self, x):
		x = torch.add(x, self.noise)
		x = self.layers(x)
		x = x.view(x.size(0), -1)
		x = self.linear(x)
		print('{:.5f}'.format(self.noise.data[0, 0, 0, 0].cpu().numpy()))
		return x


model = Net()
model.apply(utils.weights_init)
model = model.cuda()

dataset_train = getattr(datasets, 'MNIST')(root='./data', train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
dataset_test = getattr(datasets, 'MNIST')(root='./data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

optimizer = optim.SGD(model.parameters(), lr=0.001,  momentum=0.9, weight_decay=0.0001, nesterov=True)

print('\n\n****** Model Graph ******\n\n')
for arg in vars(model):
	print(arg, getattr(model, arg))

print('\n\nModel named_parameters():\n')
for name, param in model.named_parameters():
	#if param.requires_grad:
	print('{}  {}  requires_grad: {}  {:.2f}k'.format(name, list(param.size()), param.requires_grad, param.numel()/1000.))

print('\n\nModel parameters():\n')
for param in model.parameters():
	#if param.requires_grad:
	print('{} requires_grad: {}'.format(list(param.size()), param.requires_grad))

print('\n\n****** Model state_dict() ******\n\n')
for name, param in model.state_dict().items():
	print('{}  {}  requires_grad: {}'.format(name, list(param.size()), param.requires_grad))

print('\n\n')
for epoch in range(1):
	model.train()
	tr_accuracies = []
	for i, (input, label) in enumerate(loader_train):
		label = label.cuda()
		input = input.cuda()

		output = model(input)
		loss = nn.CrossEntropyLoss()(output, label)
		#print('\nBatch:', i)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		pred = output.data.max(1)[1]
		acc = pred.eq(label.data).cpu().sum() * 100.0 / 16
		tr_accuracies.append(acc)

	model.eval()
	te_accuracies = []
	with torch.no_grad():
		for i, (input, label) in enumerate(loader_test):
			label = label.cuda()
			input = input.cuda()

			output = model(input)
			pred = output.data.max(1)[1]
			acc = pred.eq(label.data).cpu().sum() * 100.0 / 16
			te_accuracies.append(acc)

	print('Epoch {:d} Train Accuracy {:.2f} Test Accuracy {:.2f}'.format(epoch, np.mean(tr_accuracies), np.mean(te_accuracies)))