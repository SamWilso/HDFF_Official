# Hyperdimensional Feature Fusion for Out-of-Distribution Detection
Hyperdimensional Feature Fusion (HDFF) is a post-hoc addition to a pretrained network that fuses multi-scale features from the network in a hyperdimensional space to enhance Out-Of-Distribution (OOD) detection performance.

This repository contains code to replicate the main results of the paper:

**[Hyperdimensional Feature Fusion for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/WACV2023/html/Wilson_Hyperdimensional_Feature_Fusion_for_Out-of-Distribution_Detection_WACV_2023_paper.html)**

*Samuel Wilson, Tobias Fischer, Niko Suenderhauf, Feras Dayoub*

Published at the 2023 IEEE/CVF Conference on Applications of Comptuer Vision (WACV).

If you find this work useful, please consider citing:
```text
@InProceedings{Wilson_2023_WACV,
    author    = {Wilson, Samuel and Fischer, Tobias and S\"underhauf, Niko and Dayoub, Feras},
    title     = {Hyperdimensional Feature Fusion for Out-of-Distribution Detection},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {2644-2654}
}
```

**Contact** 

If you have any questions, concerns or comments, please contact the first author [Sam Wilson](mailto:s84.wilson@hdr.qut.edu.au).

In the goal of producing reusable code, some aspects of this repository are still being reworked from the research codebase. In the coming weeks we will provide refactored code for: the ensembling feature monitor, MLP integration and layer-wise ablations.

## Installation
We heavily recommend using [conda](https://docs.conda.io/en/latest/) for installation.

We have included the hdff.yaml file for installing the conda environment. To create the environment, run:
```bash
conda env create -f hdff.yaml
```
Once the installation is complete, you will be able to activate the environment by running:
```bash
conda activate hdff
```
Please ensure that the hdff environment is active before attempting to run the code contained within this repository.

## Datasets
The location of the datasets is flexible with the use of the *--datadir* argument in inference.py. 

**CIFAR & MNIST & SVHN**

The CIFAR, MNIST & SVHN datasets will be automatically downloaded into the directory defined by the *--datadir* argument in inference.py. 

**TinyImageNet & LSUN**

Please download the TinyImageNet and LSUN crop + resize variants from the [ODIN](https://github.com/facebookresearch/odin) repository. Once downloaded, please rename the TinyImageNet dataset folders to "TIN_crop" and "TIN_resize" respectively, and the LSUN variants to "lsun_crop" and "lsun_resize" respectively.

**Textures & iSUN**

Please download the [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/) and [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz) (credit for providing the download link: https://github.com/wetliu/energy_ood) datasets, extracting them into your dataset folder.

## Models
**Pre-trained Models**

We provide the pretrained models we conducted our experiments on [here](https://www.dropbox.com/sh/51luvqha9qpw57d/AAAtGQ9fYF84f7y19syE0R6Xa). Please download and extract these checkpoints into the *./ckpts/* folder. Ensemble models will be available upon completion of the ensemble code. 

**Training Your Own Models**

Please refer to [this repository](https://github.com/zaeemzadeh/OOD) for training of your own models.

## Evaluation
**Running The Code**

Once everything has been setup, evaluating HDFF against all of the OOD datasets is as simple as running:
```bash
python inference.py --batch bSize --modelpath path2Model --datadir path2Data --pooling poolingType
```
where:
* `bSize` is the size of a batch as an integer. Must be greater than 1.
* `path2Model` is the path to one of the models - currently this will only target the first "Normal" or "1D" model in the directory as these were the models used to produce the results in the paper.
* `path2Data` is the path to the data directory.
* `poolingType` defines the type of pooling to be used by the feature monitor. This can be a string of either "max" or "avg".

The additional optional argument are:
* `--no-plots` disables the plotting at the end of the script.
* `--ensemble` sets HDFF to run over an ensemble of models. Currently inactive.

Once the script has completed, the results across all of the datasets will be displayed individually in the terminal and any plots will appear in the *./outs* directory if `--no-plots` was not specified.

**Using HDFF in your own code**

We have designed this repository to be flexible in nature. In order to utilise the VSA and feature monitor classes in your code, just copy the *HDFF* folder into your project. We strongly recommend using *inference.py* as a guide to see how to best utilise the feature monitor class.

In order to change the layer target of the feature monitor, we recommend defining a new layer hook function to accomodate your specific needs. As an example, here we define a search that hooks into all of the convolutional layers in the network: 
```python
class FeatureMonitor():
    ...
    def hookLayers(self):
        modules = []
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                modules.append(m)
        
	for idx, m in enumerate(modules):
		hook_fn = partial(self.__hook, idx=idx)
		m.register_forward_hook(hook_fn)
    ... 
```

