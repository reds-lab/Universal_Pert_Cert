# Towards Robustness Certification Against Universal Perturbations
![Python 3.9](https://img.shields.io/badge/python-3.9-DodgerBlue.svg?style=plastic)
![Pytorch 1.11.0](https://img.shields.io/badge/pytorch-1.11.0-DodgerBlue.svg?style=plastic)

This repository is the official implementation of the ICLR'23 paper "[Towards Robustness Certification Against Universal Perturbations](https://openreview.net/forum?id=7GEvPKxjtt)". Our goal is to provide the first practical attempt for researchers and practitioners to evaluate the robustness of their models against universal perturbations, especially to universal adversarial perturbations (UAPs) and $l_{\infty}$-norm-bounded backdoors.

## Overview

The code in this repository utilizes linear bounds calculated by [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) and further computes the certified UP robustness on a batch of data. The calculation of certified robustness can help provide robustness guarantees, identify potential weaknesses in the models and inform steps to improve their robustness.

## Requirements

+ [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) (Tested with [the February 14, 2023 version](https://github.com/Verified-Intelligence/auto_LiRPA/tree/d2592c13198e0eb536967186758c264604b59539))
+ Gurobipy >= 9.5.1

## Usage

1. Download the [example model weights](https://drive.google.com/file/d/1HACz7XpmGn7IdaS90MOg3sbg5J93J1Hz/view?usp=share_link) and extract the `./model_weights` into the same folder as the code.
2. Run Jupyter Notebooks for the demos, or load `min_correct_with_eps` from `certi_util.py` to calculate the certified UP robustness for your own model and data.

## Conclusion

We hope that this repository will serve as a valuable resource for the robustness certification community. By providing a tool to calculate the certified UP robustness, we aim to promote the development of more secure and robust machine learning models.

# Special thanks to...
[![Stargazers repo roster for @ruoxi-jia-group/Universal_Pert_Cert](https://reporoster.com/stars/ruoxi-jia-group/Universal_Pert_Cert)](https://github.com/ruoxi-jia-group/Universal_Pert_Cert/stargazers)
