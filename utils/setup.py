import torch
import argparse

## Torch Lightning Custom Datasets
from utils.datasets import CIFAR10DataModule, CIFAR100DataModule, get_transforms

def setup_args():
	## Setup the arg parser
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch', 			type=int,	default=512)
	parser.add_argument('--modelpath', 		type=str, 	default='./ckpts/WRN28_Cifar10_Normal.tar')
	parser.add_argument('--datadir', 		type=str,	default='./data/')
	parser.add_argument('--no-plots',		dest='plot', action='store_false')
	parser.add_argument('--pooling', 		type=str, 	default='max')
	parser.add_argument('--ensemble',		dest='ensemble', action='store_true')
	#parser.add_argument('--cpu',			dest='cpu', action='store_true')

	parser.set_defaults(ensemble=False)
	parser.set_defaults(plot=True)
	parser.set_defaults(cpu=False)


	return parser.parse_args()


def setup_model(args, config, device):
	#### Setup our models

	## Adjust for ensemble members
	n_ensemble = 5 if args.ensemble else 1
	
	## Selecting the correct archticture for a 1D or standard WRN
	if not 'Normal' in args.modelpath:
		from models.wideresnet_UoS import WideResNet as WRN
		model_name = './ckpts/WRN_1D_CIFAR{}_{}.tar'
	else:
		from models.wideresnet_UoS_Normal import WideResNet as WRN
		model_name = './ckpts/WRN_Normal_CIFAR{}_{}.tar'
	
	names = [model_name.format(config['n_classes'],n) for n in range(n_ensemble)]

	models = []
	for name in names:
		model = WRN(depth=28, num_classes=config['n_classes'], widen_factor=10, dropRate=0.3)
		model.load_state_dict(torch.load(name)['state_dict'])
		model.eval()
		model.to(device)
		models.append(model)

	return models


def setup_data(args, config):
	## Configure and return the datasets
	ood_names = [
		'iSUN', 'TIN_crop', 'TIN_resize', 'lsun_crop', 'lsun_resize', 
		'SVHN', 'mnist', 'kmnist', 'fasionmnist', 'dtd'
		]

	## Get the first (default) transform for our standard datasets
	transform = get_transforms()[0]

	## Setup CIFAR datasets
	c10_dm = CIFAR10DataModule(
				batch_size=args.batch,
				root=args.datadir,
				train_transforms=transform,
				val_transforms=transform
			)
	c100_dm = CIFAR100DataModule(
				batch_size=args.batch,
				root=args.datadir,
				train_transforms=transform,
				val_transforms=transform
			)
	c10_dm.setup()
	c100_dm.setup()

	if config['n_classes'] == 100:
		id_calibration_set = c100_dm.train_dataloader()
		id_test_set = c100_dm.val_dataloader()

		near_ood_test_set = c10_dm.val_dataloader()
		ood_names.append('CIFAR10')
	else:
		id_calibration_set = c10_dm.train_dataloader()
		id_test_set = c10_dm.val_dataloader()

		near_ood_test_set = c100_dm.val_dataloader()
		ood_names.append('CIFAR100')
	


	return id_calibration_set, id_test_set, near_ood_test_set, ood_names
	
