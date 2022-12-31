import torch
from tqdm import tqdm 

import torch
import torchmetrics.functional as Metrics

def inference_loop(FM, dataset, device=0) -> torch.FloatTensor:
	"""Given a the feature monitor and dataset, generates OOD scores for the dataset

	Args:
		FM (HDFF.feature_monitor.FeatureMonitor): The FeatureMonitor object pre-initialised with the target model, hooks and Feature Projection Matrices
		dataset (torch.utils.data.DataLoader): Dataloader object of the corresponding dataset
		device (int, optional): Device to run inference on. Defaults to 0.

	Returns:
		FloatTensor: OOD scores per input sample
	"""
	uncertainties = torch.empty(0).to(device)

	for x, _ in tqdm(dataset):
		## Forward pass captures features for all images in the batch
		FM.model.forward(x.to(device))
		
		## Feature monitor projects the features from each layer and then bundles them across layers
		# Result is a (batch_size, HD_space_size) tensor
		feature_bundle = FM.batchFeatureBundle()

		## Compute the cosine similarity of all feature bundles to the class bundles
		# Result is a (batch_size, n_classes) tensor
		similarity = FM.VSA.similarity(feature_bundle, FM.data['class_bundles']) # (batch, n_classes)
		
		## Retrieve the raw similarities to closest class bundle (maximum cosine similarity) 
		# Result is a (batch_size) tensor
		# values, indices = torch.max(a, dim=1) -> https://pytorch.org/docs/stable/generated/torch.max.html
		closest, _ = torch.max(similarity, dim=1)

		## Invert similarities (confidence scores) to retrieve OOD scores (uncertainties)
		## Store these OOD scores
		uncertainties = torch.cat((uncertainties, -closest))
	
	return uncertainties

def generate_metrics(ood_id, uncertainties, gt):
	"""
	Args:
		ood_id (int): The index of the target OOD dataset
		uncertainties (FloatTensor): Tensor of uncertainty scores for both ID and OOD set. Uncertainties are differentiated by 
		correpsonding index in gt
		gt (IntTensor): Tensor of indices corresponding to dataset IDs

	Returns:
		Dict: A dictionary containing the desired metrics
	"""
	## Binarise our gt to 0 for ID and 1 for OOD set
	temp_gt = torch.cat((gt[gt==0], gt[gt==ood_id]), dim=0)
	temp_gt = temp_gt > 0
	
	## Create new array of ID and target OOD set scores
	id_ucert = uncertainties[gt==0]
	ood_ucert = uncertainties[gt==ood_id]  
	temp_ucert = torch.cat((id_ucert, ood_ucert), dim=0)
	temp_ucert, temp_gt = temp_ucert.detach(), temp_gt.detach()

	prec, rec, thresh = Metrics.precision_recall_curve(temp_ucert, temp_gt, task="binary")
	
	## F1 Score
	f1 = 2 * (prec * rec) / (prec + rec)

	return f1, thresh

