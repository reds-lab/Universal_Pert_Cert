from http.client import NOT_IMPLEMENTED
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torchvision.transforms as trans
from torch.utils.data import DataLoader
import os
import torchvision.datasets as dset
from torch.utils.data import sampler



import torchvision.datasets as dset
def load_data(data_dir: str = "./data", num_imgs: int = 25, random: bool = False, dataset: str = 'MNIST') -> tuple:
    
    """
    Loads the MNIST data.
    Args:
        data_dir:
            The directory to store the full dataset.
        num_imgs:
            The number of images to extract from the test-set
        random:
            If true, random image indices are used, otherwise the first images
            are used.
    Returns:
        A tuple of tensors (images, labels).
    """

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    trns_norm = trans.ToTensor()
    
    if dataset == 'MNIST':
        cifar10_test = dset.MNIST(data_dir, train=False, download=True, transform=trns_norm)
    elif dataset == 'CIFAR':
        cifar10_test = dset.CIFAR10(data_dir, train=False, download=True, transform=trns_norm)
    else:
        raise NOT_IMPLEMENTED

    if random:
        loader_test = DataLoader(cifar10_test, batch_size=num_imgs,
                                 sampler=sampler.SubsetRandomSampler(range(10000)))
    else:
        loader_test = DataLoader(cifar10_test, batch_size=num_imgs)

    return next(iter(loader_test))

class DeltaWrapper(nn.Module):
    def __init__(self, model):
        super(DeltaWrapper, self).__init__()
        self.model = model

    def forward(self, x, delta):
        pert_x = x + delta
        return self.model(pert_x)



def min_correct_with_eps(alpha, beta, eps, label, number_class=10, verbose=False, dataset = 'MNIST'):

    if (beta.min(1)[0]>0).sum() == 0:
        print('early stop as zeros suffice')
        if dataset == 'MNIST':
            return 0, np.zeros((1,28,28))
        elif dataset == 'CIFAR':
            return 0, np.zeros((3,32,32))
        else:
            raise NOT_IMPLEMENTED
    
    #construct the MILP model
    m = gp.Model()
    m.setParam('TimeLimit', 10*60)
    if not verbose:
        m.Params.LogToConsole = 0

    c = m.addVar(lb=-10e5, ub=10e5, vtype=GRB.CONTINUOUS, name='fooled-samples')

    delta = m.addVars(alpha.shape[-1], lb=-eps, ub=eps, vtype=GRB.CONTINUOUS, name='delta')

    s = m.addVars(len(label), number_class-1, vtype=GRB.BINARY, name='s')


    q_ = m.addVars(len(label), lb=-10e5, ub=10e5, vtype=GRB.CONTINUOUS, name='q_')
    q = m.addVars(len(label), vtype=GRB.BINARY, name='q')

    aux1 = m.addVars(len(label),number_class-1,lb=-10e5, ub=10e5, vtype=GRB.CONTINUOUS, name='aux1')
    aux2 = m.addVars(len(label),number_class-1,lb=-10e5, ub=10e5, vtype=GRB.CONTINUOUS, name='aux2')
    auxsc = m.addVars(len(label), number_class-1, vtype=GRB.BINARY, name='sc')

    small_val = -10e3
    giant_val = 10e5

    #####
    m.setObjective(c, GRB.MINIMIZE)
                
    m.addConstrs(aux1[i,j] == gp.quicksum(alpha[i][j][k] * delta[k] for k in range(alpha.shape[-1])) + beta[i][j] for j in range(number_class-1)
    for i in range(len(label)))

    m.addConstrs(aux2[i,j] == aux1[i,j] * s[i, j] for j in range(number_class-1)
                                    for i in range(len(label)))

    #####---new formulation
    m.addConstr(c == gp.quicksum(q[i] for i in range(len(label))))

    m.addConstrs((q_[i] == gp.quicksum(aux2[i,j] for j in range(number_class-1)))
                for i in range(len(label)))

    m.addConstrs(giant_val*q[i] >= q_[i] for i in range(len(label)))
    m.addConstrs(giant_val*(1-q[i]) >= -q_[i] for i in range(len(label)))

    m.addConstrs(auxsc[i,j] == 1 - s[i,j] for j in range(number_class-1)
                for i in range(len(label)))

    m.addConstrs(gp.quicksum(s[i, j] for j in range(number_class-1)) == 1 for i in range(len(label)))

    m.addConstrs(aux2[i,j] + small_val *  auxsc[i, j] <= aux1[i,j]
                for j in range(number_class-1)
                for l in range(number_class-1)
                for i in range(len(label)))
    m.update()
    m.optimize()
    
    milp_delta = []
    for v in m.getVars():
        if str('delta') in v.varName:
            milp_delta.append(v.x)

    if dataset == 'MNIST':
        milp_delta = np.reshape(np.asarray(milp_delta),(1,28,28))
    elif dataset == 'CIFAR':
        milp_delta = np.reshape(np.asarray(milp_delta),(3,32,32))
    else:
        raise NOT_IMPLEMENTED

    return m.objVal, milp_delta
