# Adversarial Data Augmentation with chained transformations (Advchain)

This repo contains the pytorch implementation of adversarial data augmentation, which supports to perform adversarial training on a chain of image photometric transformations and geometric transformations for improved consistency regularization.
Please cite our work if you find it useful in your work

## Introduction

Under construction.

<!-- <img align="center" src="assets/adv_chain.png" width="750"> -->

<!-- For more details please see our [MICCAI 2020 paper](https://arxiv.org/abs/2006.13322) and [Youtube Video](https://youtu.be/-ICKhtkxY-4). -->

## Requirements

- matplotlib>=2.0
- seaborn>=0.10.0
- numpy>=1.13.3
- SimpleITK>=2.1.0
- skimage>=0.0
- torch>=1.9.0

## Set Up

1.  Install PyTorch and other required python libraries with:
    ```
    pip install -r requirements.txt
    ```
2.  Play with the provided jupyter notebook to check the enviroments

## Usage

Under construction

<!-- 1. Please ref to Sec. 4.1 and Sec 4.2 in the jupyter notebook: "adv_bias_field_generation.ipynb" to see how to plug in our module to support supervised/semi-supervised learning.
2. You can also clone this probject as submodule in your project.

- Add submodule:
  ```
  git submodule add https://github.com/cherise215/advchain.git
  ```
- Add the lib path to the file where you import our library:
  ```
  sys.path.append($change_it_to_our_project's_local_path_in_your_project$)
  ``` -->

## Citation

If you find this useful for your work, please consider citing

```
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
