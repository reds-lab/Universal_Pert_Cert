# Universal_Pert_Cert
This repo is the official implementation of the ICLR'23 paper "Towards Robustness Certification Against Universal Perturbations." We calculate the certified robustness against universal perturbations (UAP/ Backdoor) given a trained model.


The code implements linear bounds calculated using the existing method (we load <a href="https://github.com/Verified-Intelligence/auto_LiRPA">auto_LiRPA</a> as an example) to extend linear bounds to compute the certified UP robustness for a batch of data given a trained model. This repository aims to provide an accessible tool for researchers and practitioners in the field of robustness certification and machine learning security w.r.t. universal adversarial perturbations (UAPs) and $l_{\infty}$-norm bounded backdoors. 

Please download the example model weights following this link: <a href="https://drive.google.com/file/d/1HACz7XpmGn7IdaS90MOg3sbg5J93J1Hz/view?usp=share_link">Link2model_weights</a>

Please extract the ``./model_weights`` into the same folder. Also you will need to creat a ``./data`` folder to include the datasets.
