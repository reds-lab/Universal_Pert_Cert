# Towards Robustness Certification Against Universal Perturbations
![Python 3.9](https://img.shields.io/badge/python-3.9-DodgerBlue.svg?style=plastic)
![Pytorch 1.8.1](https://img.shields.io/badge/pytorch-1.8.1-DodgerBlue.svg?style=plastic)
![CUDA 10.2](https://img.shields.io/badge/cuda-10.2-DodgerBlue.svg?style=plastic)

This repository is the official implementation of the ICLR'23 paper "[Towards Robustness Certification Against Universal Perturbations](https://openreview.net/forum?id=7GEvPKxjtt)". Our goal is to provide the first practical attempt for researchers and practitioners to evaluate the robustness of their models against universal perturbations, especially to universal adversarial perturbations (UAPs) and $l_{\infty}$-norm-bounded backdoors.

## Overview
The code in this repository implements linear bounds calculated using the existing method, [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA), to extend linear bounds and compute the certified UP robustness for a batch of data given a trained model. The calculation of certified robustness can help identify potential weaknesses in the models and inform steps to improve their robustness.

## TO-DO
- Example model weights (can be downloaded from [Link2model_weights](https://drive.google.com/file/d/1HACz7XpmGn7IdaS90MOg3sbg5J93J1Hz/view?usp=share_link))
- Datasets to be placed in a `./data` folder

## Requirements
+ Python >= 3.9.6
+ PyTorch >= 1.8.1+cu102
+ TorchVisison >= 0.9.1+cu102
+ Gurobipy >= 9.5.1
+ [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)

## Usage
1. Download the example model weights and extract the `./model_weights` into the same folder as the code. 
2. Create a `./data` folder and place the datasets inside. 
3. You can also load `min_correct_with_eps` from `certi_util.py` to calculate the certified UP robustness for your trained model and data.

## Conclusion
We hope that this repository will serve as a valuable resource for the robustness certification community. By providing a tool to calculate the certified UP robustness, we aim to promote the development of more secure and robust machine learning models.


# Special thanks to...
[![Stargazers repo roster for @ruoxi-jia-group/Universal_Pert_Cert](https://reporoster.com/stars/ruoxi-jia-group/Universal_Pert_Cert)](https://github.com/ruoxi-jia-group/Universal_Pert_Cert/stargazers)
