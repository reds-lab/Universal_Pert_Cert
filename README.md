# Towards Robustness Certification Against Universal Perturbations

This repository is the official implementation of the ICLR'23 paper with the same title. Our goal is to provide the first practical attempt for researchers and practitioners to evaluate the robustness of their models against universal perturbations, especially to universal adversarial perturbations (UAPs) and $l_{\infty}$-norm-bounded backdoors.

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

## Usage
1. Download the example model weights and extract the `./model_weights` into the same folder as the code. 
2. Create a `./data` folder and place the datasets inside. 
3. You can also load `min_correct_with_eps` from `certi_util.py` to calculate the certified UP robustness for your trained model and data.

## Conclusion
We hope that this repository will serve as a valuable resource for the robustness certification community. By providing a tool to calculate the certified UP robustness, we aim to promote the development of more secure and robust machine learning models.
