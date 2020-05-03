#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from fedl.servers.serveravg import FedAvg
from fedl.servers.serverapfl import APFL
from fedl.servers.serverpsnl import Persionalized
from fedl.servers.serverperavg import PerAvg
from fedl.trainmodel.models import Mclr_Logistic, Net, Mclr_CrossEntropy, DNN
from utils.plot_utils import plot_summary_one_figure_mnist
import torch
torch.manual_seed(0)

# plot for synthetic
numusers = 5
num_glob_iters = 800
dataset = "Mnist"

# --------------------------------------------Non Convex Case----------------------------------------#
if(1):
    local_ep = [20,20,20,20,20,20]
    lamda = [15,15,15,15,15,15]
    learning_rate = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
    alpha =  [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    batch_size = [20,20,20,20,20,20]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.09,0.09,0.09,0.09] 
    algorithms = [ "Persionalized_p","Persionalized","FedAvg","PerAvg_p"]
    plot_summary_one_figure_mnist(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, alpha = alpha, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

# -------------------------------------------- Convex Case----------------------------------------#
if(0):
    local_ep = [20,20,20,20,20,20]
    lamda = [15,15,15,15,15,15]
    learning_rate = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    alpha =  [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    batch_size = [20,20,20,20,20,20]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.009,0.009,0.009,0.009] 
    algorithms = [ "Persionalized_p","Persionalized","FedAvg","PerAvg_p"]
    plot_summary_one_figure_mnist(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, alpha = alpha, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
