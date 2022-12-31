import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets as D
from torchvision.datasets import SVHN, MNIST, KMNIST, FashionMNIST, ImageFolder

class CIFAR10DataModule(pl.LightningDataModule):
	def __init__(self, batch_size=32, root='./',  train_transforms=None, val_transforms=None, num_workers=1):
		super().__init__()
		self.batch_size=batch_size
		self.root = root
		self.train_transforms = train_transforms
		self.val_transforms = val_transforms
		self.num_workers = num_workers

	def setup(self, *args, **kwargs):
		self.train_set = D.CIFAR10(root=self.root, train=True, transform=self.train_transforms, download=True)
		self.val_set = D.CIFAR10(root=self.root, train=False, transform=self.val_transforms, download=True)

	def train_dataloader(self):
		train_data = DataLoader(
			self.train_set,
			batch_size=self.batch_size,
			shuffle=True,
			drop_last=True,
			num_workers=self.num_workers
		)
		
		return train_data

	def val_dataloader(self):
		val_data = DataLoader(
			self.val_set,
			batch_size=self.batch_size,
			shuffle=False,
			drop_last=False,
			num_workers=self.num_workers
		)

		return val_data
	
	def prepare_data(self):
		pass

class CIFAR100DataModule(pl.LightningDataModule):
	def __init__(self, batch_size=32, root='./',  train_transforms=None, val_transforms=None, num_workers=1):
		super().__init__()
		self.batch_size=batch_size
		self.root = root
		self.train_transforms = train_transforms
		self.val_transforms = val_transforms
		self.num_workers = num_workers

	def setup(self, *args, **kwargs):
		self.train_set = D.CIFAR100(root=self.root, train=True, transform=self.train_transforms, download=True)
		self.val_set = D.CIFAR100(root=self.root, train=False, transform=self.val_transforms, download=True)

	def train_dataloader(self):
		train_data = DataLoader(
			self.train_set,
			batch_size=self.batch_size,
			shuffle=True,
			drop_last=True,
			num_workers=self.num_workers
		)
		
		return train_data

	def val_dataloader(self):
		val_data = DataLoader(
			self.val_set,
			batch_size=self.batch_size,
			shuffle=False,
			drop_last=True,
			num_workers=self.num_workers
		)

		return val_data

	def prepare_data(self):
		pass



def get_transforms():
	import torchvision.transforms as T
	norm = T.Normalize(
		mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
		std=[x/255.0 for x in [63.0, 62.1, 66.7]]
	)

	default_transform = T.Compose([
		T.CenterCrop(size=(32, 32)),
		T.ToTensor(),
		norm
	])

	## Custom Transforms for SVHN, ISUN, Textures and MNIST following with:
	# Lin et al. (2022) "MOOD: Multi-level Out-of-Distribution Detection" (Appendix A)
	svhn_isun_textures_transform = T.Compose([
		T.Resize(32),
		T.CenterCrop(size=(32, 32)),
		T.ToTensor(),
		norm
	])

	mnist_transform = T.Compose([
		T.Grayscale(num_output_channels=3),
		T.Pad(padding=2),
		T.ToTensor(),
		norm
	])

	return default_transform, svhn_isun_textures_transform, mnist_transform

def get_ood_dataset(ood, batch):
	default_transform, svhn_isun_textures_transform, mnist_transform = get_transforms()

	if ood == 'SVHN':
		OOD_set = SVHN(root='./data/', split='test', transform=svhn_isun_textures_transform, download=True)
		OOD_set = DataLoader(OOD_set, batch_size=batch, num_workers=1)
	elif ood == 'fasionmnist':
		OOD_set = FashionMNIST(root=f'./data/{ood}'.format(ood), train=False, transform=mnist_transform, download=True)
		OOD_set = DataLoader(OOD_set, batch_size=batch, num_workers=1)
	elif ood == 'mnist':
		OOD_set = MNIST(root=f'./data/{ood}', train=False, transform=mnist_transform, download=True)
		OOD_set = DataLoader(OOD_set, batch_size=batch, num_workers=1)
	elif ood == 'kmnist':
		OOD_set = KMNIST(root=f'./data/{ood}'.format(ood), train=False, transform=mnist_transform, download=True)
		OOD_set = DataLoader(OOD_set, batch_size=batch, num_workers=1)
	else:
		if ood == 'dtd':
			ood = 'dtd/images'
			transform = svhn_isun_textures_transform
		elif ood == 'iSUN':
			transform = svhn_isun_textures_transform
		else:
			transform = default_transform
		ood_path = f'./data/{ood}'
		OOD_set = ImageFolder(ood_path, transform=transform)
		OOD_set = DataLoader(OOD_set, batch_size=batch, num_workers=1)
	return OOD_set