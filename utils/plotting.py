import torch
import matplotlib.pyplot as plt
import seaborn as sns

from utils.eval import inference_loop, generate_metrics

sns.set(font_scale=2, rc={'figure.figsize':(18, 5)})
sns.set_style('whitegrid')

def plot_f1(OODs, uncertainties, labels, id_name):
	"""
	Plots the F1 score for each OOD dataset as a function of the threshold, highlighting the optimal region for all far-OOD datasets.
	Replicates the experiments from Figure 2 in the paper.

	Args:
		OODs (List[String]): List of OOD dataset names.
		uncertainties (torch.FloatTensor): OOD scores for previous datasets.
		labels (torch.FloatTensor): Labels for the OOD scores. 0 for ID test and 1+ for OODs. Used to index into uncertainties.
		id_name (String): Name of the in-distribution dataset
	"""
	## Clear figure just to be sure
	plt.clf()

	## Dictionary makes it a little easier to keep track of the min and max
	optimal_region = {
		'max': 90,
		'min': 0
	}

	## Repeat for each OOD dataset
	for ood, ood_id in zip(OODs, torch.arange(len(OODs))+1):
		## Generate the F1 and Threshold metrics
		f1, thresh = generate_metrics(ood_id, uncertainties, labels)
		
		## Convert the threshold to degrees
		#thresh = torch.rad2deg(torch.arccos(thresh * -1))
		thresh = torch.cat((torch.Tensor([thresh[0]]).cuda(), thresh))
		
		## Error checking for NaNs
		thresh = thresh[~torch.isnan(f1)]
		f1 = f1[~torch.isnan(f1)]

		print(f1, thresh)## Ignore near-OOD setting
		if not 'CIFAR' in ood:
			## Find the acceptable (within 5% of the best F1 score) range of thresholds
			mask = f1 > f1.max()-0.05
			upper, lower = thresh[mask].max().item(), thresh[mask].min().item()

			## Shrink the region (lower the max, increase the minimum) to accomodate the new dataset 
			optimal_region['max'] = min(upper, optimal_region['max'])
			optimal_region['min'] = max(lower, optimal_region['min']) 

		f1 = f1.detach().cpu().numpy()
		thresh = thresh.detach().cpu().numpy()
		lineplot = sns.lineplot(
			x=thresh,
			y=f1,
			label=OODs[ood_id-1],
			#color=colours[ood_id-1],
			linewidth=2
		)
	
	## Generate the figure
	fig = lineplot.get_figure()
	lineplot.legend(loc='lower left')

	## Axis titles
	lineplot.set_xlabel('Critical Threshold (Degrees)')
	lineplot.set_ylabel('F1 Score')

	## Set limit (solely for aesthetics)
	lineplot.set_xlim(0, 70)

	## Colour the optimal region for all far-OOD datasets
	plt.axvspan(optimal_region['max'], optimal_region['min'], facecolor='lightgray')
	fig.savefig(f'./outs/F1_critical_thresh_{id_name}.pdf', bbox_inches='tight')
	fig.clf()


def plot_angles(id_name, calibration_set, OODs, FM, uncertainties, labels):
	"""
	Plots the distribution of the angles to the class bundles for the ID test set, ID training set and OODs.
	Replicates the experiment in Figure 4 of the paper.

	Args:
		id_name (String): Name of the in-distribution dataset.
		calibration_set (torch.utils.data.DataLoader): The training ID set.
		OODs (List[String]): List of OOD dataset names.
		FM (HDFF.feature_monitor.FeatureMonitor): FeatureMonitor object used to collect the scores for the previous datasets.
		uncertainties (torch.FloatTensor): OOD scores for previous datasets.
		labels (torch.FloatTensor): Labels for the OOD scores. 0 for ID test and 1+ for OODs. 
	"""
	## Clear figure just to be sure
	plt.clf()

	## Set up plotting names and colours
	names = [f'{id_name.upper()} (test)', f'{id_name.upper()} (train)']
	colors = ['blue', 'green', 'black', 'orange', 'red', 'tab:brown']
	names.extend(OODs)
	
	## Reassign labels to be 0 for ID test, 1 for ID train and 2+ for OODs
	labels[labels > 0] += 1
	
	## Capture the ID training set angles
	ucert = inference_loop(FM, calibration_set)
	uncertainties = torch.cat((uncertainties, ucert), dim=0)
	labels = torch.cat((labels, torch.ones(len(ucert)).to(0)))

	for idx in torch.unique(labels):
		## Capture relevant uncertainties
		uncertainty_subset = uncertainties[labels == idx]
		
		## Convert to angles
		angles = torch.rad2deg(torch.arccos(uncertainty_subset * -1))
		
		## Plot the distribution of the angles
		sns.kdeplot(
			angles.detach().cpu().numpy(),
			color=colors[int(idx.item())],
			label=names[int(idx.item())],
			linewidth=3
		)
	
	## Axis labels
	plt.xlabel('Angular Distance (Degrees)')
	plt.ylabel('Normalised Count (%)')
	
	## Set lower limit (angle cannot be less than 0)
	plt.xlim(0)
	plt.ylim(0)

	## Add legend and save
	plt.legend()
	plt.savefig(f'./outs/Dist_{id_name}.pdf', bbox_inches='tight')
