# Adversarial Data Augmentation with Chained Transformations (Adv Chain)

This repo contains the pytorch implementation of adversarial data augmentation, which supports to perform adversarial training on a chain of image photometric transformations and geometric transformations for improved consistency regularization.
Please cite our work if you find it useful in your work.

## License:
All rights reserved. 

## Introduction

Adv Chain takes both image information and network's current knowledge into account, and utilizes these information to find effective transformation parameters that are beneficial for the downstream segmentation task. Specifically, the underlying image transformation parameters are optimized so that the dissimilarity/inconsistency between the network's output for clean data and the output for perturbed/augmented data is maximized.

<img align="center" src="assets/graphical_abstract.png" width="750">

As shown below, the learned adversarial data augmentation focuses more on deforming/attacking region of interest, generating realistic adversarial examples that the network is sensitive at. In our experiments, we found that augmenting the training data with these adversarial examples are beneficial for enhancing the segmentation network's generalizability.
<img align="center" src="assets/cardiac_example.png" width="750">

For more details please see our paper on [arXiv](https://arxiv.org/abs/2108.03429).

## Requirements

- matplotlib>=2.0
- seaborn>=0.10.0
- numpy>=1.13.3
- SimpleITK>=2.1.0
- skimage>=0.0
- torch>=1.9.0

## Set Up
1.  Upgrade pip to the latest:
    ```
    pip install --upgrade pip
    ```
1.  Install PyTorch and other required python libraries with:
    ```
    pip install -r requirements.txt
    ```
2.  Play with the provided jupyter notebook to check the enviroments, see `example/adv_chain_data_generation_cardiac.ipynb` to find example usage.

## Usage

1. You can clone this probject as submodule in your project.

- Add submodule:
  ```
  git submodule add https://github.com/cherise215/advchain.git
  ```
- Add the lib path to the file where you import our library:
  ```
  sys.path.append($path-to-advchain$)
  ```

2. Import the library and then add it to your training codebase. Please refer to examples under the `example/` folder for more details.



## News:
[2022-07-16] now support 3D augmentation (beta)! Please see `advchain/example/adv_chain_data_generation_cardiac.ipynb` to find example usage.

## Guide:
1. Please perform adversarial data augmentation *before* computing standard supervised loss
2. For networks with dropout layers, please replace 'nn.Dropout2d' or  nn.Dropout3d with fixable dropout layers to allow optimization with fixed network structure. We provide 2D, and 3D fixable dropout layers in "advchain.common.layers.Fixable2DDropout, advchain.common.layers.Fixable3DDropout". 
3. for semi-supervised learning, please perform adversarial data augmentation on labelled and unlabelled batch *separately*. 


## Citation

If you find this useful for your work, please consider citing

```
@ARTICLE{Chen_2021_Enhancing,
  title  = "Enhancing {MR} Image Segmentation with Realistic Adversarial Data Augmentation",
  journal = {arXiv Preprint},
  author = "Chen, Chen and Qin, Chen and Ouyang, Cheng and Wang, Shuo and Qiu,
            Huaqi and Chen, Liang and Tarroni, Giacomo and Bai, Wenjia and
            Rueckert, Daniel",
    year = 2021,
    note = {\url{https://arxiv.org/abs/2108.03429}}
}


@INPROCEEDINGS{Chen_MICCAI_2020_Realistic,
  title     = "Realistic Adversarial Data Augmentation for {MR} Image
               Segmentation",
  booktitle = "Medical Image Computing and Computer Assisted Intervention --
               {MICCAI} 2020",
  author    = "Chen, Chen and Qin, Chen and Qiu, Huaqi and Ouyang, Cheng and
               Wang, Shuo and Chen, Liang and Tarroni, Giacomo and Bai, Wenjia
               and Rueckert, Daniel",
  publisher = "Springer International Publishing",
  pages     = "667--677",
  year      =  2020
}

```
