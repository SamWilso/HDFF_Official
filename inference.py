## Non-HDFF Imports
import torch
import utils.calculate_log as callog

## HDFF Imports
from HDFF.VSAs import HDFF_VSA
from HDFF.feature_monitor import FeatureMonitor

## Eval imports 
from utils.eval import inference_loop

## Datasets
from utils.datasets import get_ood_dataset

## Setup Imports 
from utils.setup import setup_args, setup_model, setup_data

## Plotting imports 
from utils.plotting import plot_f1, plot_angles

def main(args, config):
    device = 0 if not args.cpu else 'cpu'
    ## Setup our models
    models = setup_model(args, config, device=device)

    ## Setup the VSA & Feature Monitor instances
    VSA = HDFF_VSA()
    FM = FeatureMonitor(models, VSA, config, device=device)

    ## Hook into the layers & projections matrices; preparing for inference
    FM.hookLayers()

    ## Setup the data
    id_calibration_set, id_test_set, near_ood_test_set, ood_names = setup_data(args, config)

    ## We label OODness scores as 0 for ID and > 0 for OOD
    id_label = 0
    ood_labels = torch.arange(len(ood_names)) + 1

    FM.captureFeatureMeans(id_calibration_set)
    FM.createClassBundles(id_calibration_set)

    ## Inference
    # First on the ID test set
    scores = inference_loop(FM, id_test_set)
    labels = torch.zeros_like(scores)

    ## Then on the OOD test sets
    for ood_name, ood_label in zip(ood_names, ood_labels):
        print(f'Capturing scores for {ood_name}')
        if 'CIFAR' in ood_name:
            ood_set = near_ood_test_set
        else:
            ood_set = get_ood_dataset(ood_name, args.batch)
        temp_scores = inference_loop(FM, ood_set)
        scores = torch.cat([scores, temp_scores])
        labels = torch.cat([labels, ood_label * torch.ones_like(temp_scores)])

    ## Calculate and present the results for each datasets
    id_scores = scores[labels == id_label]
    for ood_name, ood_label in zip(ood_names, ood_labels):
        ood_scores = scores[labels == ood_label]
        results = callog.compute_metric(
            -id_scores.detach().cpu().numpy(),
            -ood_scores.detach().cpu().numpy()
        )
        print(f'Metrics for {ood_name}')
        callog.print_results(results)

    if args.plot:
        ## Aggregate over the MNIST, SUN and Other OODs
        names = ['MNIST (AVG)', 'SUN (AVG)', 'Other (AVG)', ood_names[-1]]
        id_lists = [[7, 8, 9], [1, 4, 5], [2, 3, 6, 10]]
        new_labels = labels.clone()
        for index, ids in enumerate(id_lists):
            mask = torch.isin(labels, torch.tensor(ids).to(device))
            new_labels[mask] = index + 1
        labels = new_labels

        ## Replace the label for near-OOD
        labels[labels == labels.max()] = 4

        ## Plot the F1 Scores
        ## Currently disabled - behaviour of some libaries has changed since the paper was written
        ## TODO: Readd this functionality with the new behaviour
        #plot_f1(names, scores, labels, config['name'])

        ## Plot the Angles
        plot_angles(config['name'], id_calibration_set, names, FM, scores, labels)

        exit()


if __name__ == '__main__':
    ## Retrieve our arguments
        args = setup_args()

        ## Setup defaults & config from args
        n_classes = 100 if '100' in args.modelpath else 10
        data = f'cifar{n_classes}'
        config = {
                'hyper_size': int(1e4),
                'n_classes': n_classes,
                'model': args.modelpath,
                'input_size': (32, 32),
                'pool': args.pooling,
                'name': data
                }

        main(args, config)

