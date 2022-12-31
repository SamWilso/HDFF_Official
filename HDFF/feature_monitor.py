import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm


from models.wideresnet_UoS import BasicBlock, NetworkBlock
from models.wideresnet_UoS_Normal import BasicBlock as BasicBlock2, NetworkBlock as NetworkBlock2
		

from functools import partial

from utils.transforms import identity, mean_centre

class FeatureMonitor():
	def __init__(self, models, VSA, config, device='cpu'):
		self.model		= models[0] ## Temporarily only support single model
		self.VSA		= VSA
		self.device 	= device
		self.config		= config
		self.data		= {}
		self.pool 		= {
			'avg': F.avg_pool2d,
			'max': F.max_pool2d,
		}[config['pool']]
		self.preprocess = identity


		self.model.eval()
		self.model.to(self.device)

	@torch.no_grad()
	def __hook(self, model_self, inputs, outputs, idx):
		"""
		Hook function to store the output of each layer
		"""
		self.features[idx] = outputs.clone()
		
	def hookLayers(self):
		"""
		Searches through all of the modules and applies hooks to the critical layers of interest
		"""
		layer_count = 0
		modules = []

		## Search through all the modules
		for m in self.model.modules():

			## If we find a BasicBlock we need to search through the submodules
			if isinstance(m, BasicBlock) or isinstance(m, BasicBlock2):

				## Capture any Conv2d, ReLU, or Identity layers
				for n in m.modules():	
					if isinstance(n, nn.Conv2d):
						name = 'Conv2d'
					elif isinstance(n, nn.ReLU):
						name = 'ReLu'
					elif isinstance(n, nn.Identity):
						name = 'SkipConnection'
					else: 
						continue
					print(f'Adding BasicBlock {name} to hook queue. Hook index: {layer_count}')
					modules.append(n)
					layer_count += 1
				
				## Add the output of the BasicBlock to the hook queue
				print(f'Adding BasicBlock output to hook queue. Hook index: {layer_count}')
				modules.append(m)
				layer_count += 1

				## Add the outputs of the NetworkBlocks to the hook queue
			elif isinstance(m, NetworkBlock) or isinstance(m, NetworkBlock2):
				print(f'Adding NetworkBlock output to hook queue. Hook index: {layer_count}')
				modules.append(m)
				layer_count += 1
	
		## Optionally add layer filtering here
		# e.g modules = [m for m in modules if m.out_channels == 64]

		self.n_hooks = len(modules)
		self.features = [0] * self.n_hooks
		self.hook_tracker = 0

		## Apply hook function to targetted modules
		for idx, m in enumerate(modules):
			## "partial" allows us to pass the index of the layer to the hook function
			hook_fn = partial(self.__hook, idx=idx)
			m.register_forward_hook(hook_fn)
		
		## Sample the hyperdimensional projection matrices for each layer
		self.sampleProjectionMatrices()

	@torch.no_grad()
	def batchFeatureBundle(self):
		"""
		Layer-wise bundling of features for a batch of images. Result is a single vector for each image in the batch.
		
		Returns:
			feature_bundle (torch.FloatTensor): the image descriptor vector for each image in the batch
		"""
		
		## For each layer, apply mean centering, pooling, and projection
		for idx, feat in enumerate(self.features):
			## Apply mean centering (if applicable)
			feat = self.preprocess(feat, self.means[idx])
			## Apply pooling
			feat = self.pool(feat, kernel_size=feat.size()[2:]).squeeze() # (batch_size, n_channels)
			## Project into high dimensional space
			feat @= self.proj[idx].to(self.device)# (batch_size, hyper_dim)
			## Store if first batch else apply bundling
			feature_bundle = feat if not idx else self.VSA.bundle(feature_bundle, feat)
		
		return feature_bundle

	@torch.no_grad()
	def captureFeatureMeans(self, calibration_set):
		"""
		Captures the means of feature activations over the calibration set for each layer targetted

		Args:
			calibration_set (Dataloader): Dataloader object corresponding to the in-distribution calibration set
		"""
		## Define a count and a list to store the means
		means = []
		count = 0

		for batch_idx, (x, _) in enumerate(tqdm(calibration_set)):
			count += len(x)
			self.model.forward(x.to(self.device))

			## First batch we need to initialize the means
			if not batch_idx:
				for feat in self.features:
					_, n_channels, _, _ = feat.size()
					means.append(torch.zeros((1, n_channels)).to(self.device))
			
			## Sum the pooled feature activations for each layer
			for idx, feat in enumerate(self.features):
				feat = F.avg_pool2d(feat, kernel_size=feat.size()[2:]).squeeze() # (batch_size, n_channels)
				means[idx] = torch.sum(
					torch.cat((means[idx], feat), dim=0),
					dim=0,
					keepdim=True
				)

		## Calculate a single mean value of the pooled feature activations for each layer
		self.means = torch.zeros(len(means))
		for idx, m in enumerate(means):
			self.means[idx] = torch.mean(m/count)
			
		## Set the preprocessing function to mean centering
		self.preprocess = mean_centre
		

	@torch.no_grad()
	def createClassBundles(self, calibration_set):
		"""
		Creates the class descriptor vectors for each class in the calibration set

		Args:
			calibration_set (torchvision.Dataset): The known in-distribution calibration set
		"""
		## Initialise the class bundles
		bundles = torch.zeros(
			(self.config['n_classes'], 
			self.config['hyper_size'])
			).to(self.device)
		
		## Iterate over the calibration set
		for x, y in tqdm(calibration_set):
			y = y.squeeze().to(self.device)
			self.model.forward(x.to(self.device))
			## Collect image descriptor bundles
			feature_bundle = self.batchFeatureBundle()
			## Bundle the image descriptors for each class
			for label in torch.unique(y):
				## Get the indices of the images in the batch corresponding to the current class
				mask = label == y
				## Use the VSA to bundle the image descriptors corresponding the current class
				bundled_rep = self.VSA.bulk_bundle(feature_bundle[mask])
				## Iteratively update the class descriptor vector
				bundles[label] = self.VSA.bundle(bundled_rep, bundles[label])
		
		## Store the class descriptor vectors
		self.data['class_bundles'] = bundles

	@torch.no_grad()
	def sampleProjectionMatrices(self):
		"""
		Generates the hyperdimensional projection matrices for each layer
		"""
		## Dummy input for forward pass
		sample_input = torch.zeros((1, 3) + self.config['input_size'])
		self.model.forward(sample_input.to(self.device))
		matrices = []
		for f in self.features:
			## Grab number of channels for feature map
			c = f.size()[1]
			## Sample projection matrix
			proj_matrix = torch.empty((c, self.config['hyper_size']))
			## Initialise projection matrix with orthogonal values
			nn.init.orthogonal_(proj_matrix)
			## Store projection matrix
			matrices.append(proj_matrix)
		## Save into feature monitor memory
		self.proj = matrices
