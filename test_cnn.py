import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid



dataset_url = "http://files.fast.ai/data/cifar10.tgz"
download_url(dataset_url, '.')

with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
	tar.extractall(path='./data')


data_dir = './data/cifar10'

classes = os.listdir(data_dir, "/train")

dataset = ImageFolder(data_dir+'/train', transform=ToTensor())


def show_example(img, label):
	print('Label: ', dataset.classes[label], "("+str(label)+")")
	plt.imshow(img.permute(1, 2, 0))


random_seed = 42
torch.manual_seed(random_seed)

val_size = 5000
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])


batch_size = 128

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)


def show_batch(dl):
	for images, labels in dl:
		fig, ax = plt.subplots(figsize=(12, 6))
		ax.set_xticks([]); ax.set_yticks([])
		ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
		break


def apply_kernel(image, kernel):
	ri, ci = image.shape
	rk, ck = kernel.shape
	ro, co = ri-rk+1, ci-ck+1
	output = torch.zeros([ro, co])
	for i in range(ro):
		for j in range(co):
			output[i, j] = torch.sum(image[i:i+rk, j:j+ck] * kernel)
	return output


def accuracy(outputs, labels):
	_, preds = torch.max(outputs, dim=1)
	return torch.tensor(torch.sum(preds == labels).item() / len(preds))





class ImageClassificationBase(nn.Module):
	def training_step(self, batch):
		images, labels = batch
		out = self(images)
		loss = F.cross_entropy(out, labels)
		return loss

	def validation_step(self, batch):
		images, labels = batch
		out = self(images)
		loss = F.cross_entropy(out, labels)
		acc = accuracy(out, labels)
		return {'val_loss': loss, 'val_acc': acc}

	def validation_epoch_end(self, outputs):
		batch_losses = [x['val_loss'] for x in outputs]
		epoch_loss = torch.stack(batch_losses).mean()
		batch_accs = [x['val_acc'] for x in outputs]
		epoch_acc = torch.stack(batch_accs).mean()
		return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

	def epoch_end(self, epoch, result):
		print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))




class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.network = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),   # Output: 64 x 16 x 16

			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),   # Output: 128 x 8 x 8

			nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),   # Output: 256 x 4 x 4

			nn.Flatten(),
			nn.Linear(256*4*4, 1024),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Dropout(0.4),
			nn.Linear(512, 10))

	def forward(self, xb):
		return self.network(xb)





@torch.no_grad()
def evaluate(model, val_loader):
	model.eval()
	outputs = [model.validation_step(batch) for batch in val_loader]
	return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
	history = []
	optimizer = opt_func(model.parameters(), lr)
	for epoch in range(epochs):
		# Training phase
		model.train()
		train_losses = []
		for batch in train_loader:
			loss = model.training_step(batch)
			train_losses.append(loss)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		# Validation phase
		result = evaluate(model, val_loader)
		result['train_loss'] = torch.stack(train_losses).mean().item()
		model.epoch_end(epoch, result)
		history.append(result)
	return history




model = Model()

evaluate(model, val_dl)

num_epochs = 10
opt_func = torch.optim.Adam
lr = 0.001


history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

test_dataset = ImageFolder(data_dir+'/test', transform=ToTensor())

def predict_image(img, model):
	xb = img.unsqueeze(0)
	yb = model(xb)
	_, preds = torch.max(yb, dim=1)
	return dataset.classes[preds[0].item()]


img, label = test_dataset[0]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))


# test_loader = DataLoader(test_dataset, batch_size*2)
# result = evaluate(model, test_loader)

# torch.save(model.state_dict(), 'cnn_model.pth')