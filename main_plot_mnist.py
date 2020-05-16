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
from utils.plot_utils import plot_summary_one_figure_mnist_L,plot_summary_one_figure_mnist_D,plot_summary_one_figure_mnist_R,plot_summary_one_figure_mnist_K,plot_summary_one_figure_mnist_Compare,plot_summary_one_figure_mnist_Beta
import torch
torch.manual_seed(0)

# plot for synthetic
numusers = 5
num_glob_iters = 600
dataset = "Mnist"

# --------------------------------------------Non Convex Case----------------------------------------#
if(0): # L values
    local_ep = [20,20,20,20,20,20,20]
    lamda = [7,10,15,7,10,15]
    learning_rate = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
    alpha =  [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    batch_size = [20,20,20,20,20,20]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.09,0.09,0.09,0.09,0.09,0.09] 
    algorithms = [ "Persionalized_p","Persionalized_p","Persionalized_p","Persionalized","Persionalized","Persionalized"]
    plot_summary_one_figure_mnist_L(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, alpha = alpha, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

if(0):  # R values
    local_ep = [10,20,30,10,20,30]
    lamda = [15,15,15,15,15,15,15]
    learning_rate = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
    alpha =  [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    batch_size = [20,20,20,20,20,20]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.09,0.09,0.09,0.09,0.09,0.09] 
    algorithms = [ "Persionalized_p","Persionalized_p","Persionalized_p","Persionalized","Persionalized","Persionalized"]
    plot_summary_one_figure_mnist_R(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, alpha = alpha, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

if(0): # D values
    local_ep = [20,20,20,20,20,20,20]
    lamda = [15,15,15,15,15,15,15]
    learning_rate = [0.004, 0.005, 0.005, 0.004, 0.005, 0.005]
    alpha =  [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    batch_size = [10,20,50,10,20,50]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.08,0.09,0.09,0.08,0.09,0.09] 
    algorithms = [ "Persionalized_p","Persionalized_p","Persionalized_p","Persionalized","Persionalized","Persionalized"]
    plot_summary_one_figure_mnist_D(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, alpha = alpha, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

if(0): # K value
    local_ep = [20,20,20,20,20,20,20]
    lamda = [15,15,15,15,15,15,15]
    learning_rate = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
    alpha =  [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    batch_size = [20,20,20,20,20,20]
    K = [1,5,7,1,5,7]
    personal_learning_rate = [0.09,0.09,0.09,0.09,0.09,0.09] 
    algorithms = [ "Persionalized_p","Persionalized_p","Persionalized_p","Persionalized","Persionalized","Persionalized"]
    plot_summary_one_figure_mnist_K(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, alpha = alpha, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

if(0): # comparision 
    local_ep = [20,20,20,20,20,20,20]
    lamda = [30,30,15,15,15]
    learning_rate = [0.005, 0.005, 0.005, 0.005, 0.005]
    alpha =  [0.001, 0.001, 0.001, 0.001, 0.001]
    batch_size = [20,20,20,20,20]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.05,0.05,0.09,0.09,0.09] 
    algorithms = [ "Persionalized_p","Persionalized","PerAvg_p","FedAvg"]
    plot_summary_one_figure_mnist_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, alpha = alpha, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)


# -------------------------------------------- Convex Case----------------------------------------#
if(0): # L values
    local_ep = [20,20,20,20,20,20,20]
    lamda = [7,10,15,7,10,15]
    learning_rate = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
    alpha =  [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    batch_size = [20,20,20,20,20,20]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.09,0.09,0.09,0.09,0.09,0.09] 
    algorithms = [ "Persionalized_p","Persionalized_p","Persionalized_p","Persionalized","Persionalized","Persionalized"]
    plot_summary_one_figure_mnist_L(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, alpha = alpha, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
if(0):  # R values
    local_ep = [10,20,30,10,20,30]
    lamda = [15,15,15,15,15,15,15]
    learning_rate = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
    alpha =  [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    batch_size = [20,20,20,20,20,20]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.09,0.09,0.09,0.09,0.09,0.09] 
    algorithms = [ "Persionalized_p","Persionalized_p","Persionalized_p","Persionalized","Persionalized","Persionalized"]
    plot_summary_one_figure_mnist_R(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, alpha = alpha, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
if(0): # D values
    local_ep = [20,20,20,20,20,20,20]
    lamda = [15,15,15,15,15,15,15]
    learning_rate = [0.004, 0.005, 0.005, 0.004, 0.005, 0.005]
    alpha =  [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    batch_size = [10,20,50,10,20,50]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.08,0.09,0.08,0.08,0.09,0.09] 
    algorithms = [ "Persionalized_p","Persionalized_p","Persionalized_p","Persionalized","Persionalized","Persionalized"]
    plot_summary_one_figure_mnist_D(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, alpha = alpha, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
if(0): # K value
    local_ep = [20,20,20,20,20,20,20]
    lamda = [15,15,15,15,15,15,15]
    learning_rate = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
    alpha =  [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    batch_size = [20,20,20,20,20,20]
    K = [1,5,7,1,5,7]
    personal_learning_rate = [0.09,0.09,0.09,0.09,0.09,0.09] 
    algorithms = [ "Persionalized_p","Persionalized_p","Persionalized_p","Persionalized","Persionalized","Persionalized"]
    plot_summary_one_figure_mnist_K(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, alpha = alpha, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
if(0): # comparision
    local_ep = [20,20,20,20,20,20,20]
    lamda = [15,15,15,15]
    learning_rate = [0.005, 0.005, 0.005, 0.005]
    alpha =  [0.001, 0.001, 0.001, 0.001]
    batch_size = [20,20,20,20]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.1,0.1,0.09,0.09,0.09,0.09]
    algorithms = [ "Persionalized_p","Persionalized","PerAvg_p","FedAvg"]
    plot_summary_one_figure_mnist_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, alpha = alpha, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)



if(1): # Beta value
    local_ep = [20,20,20,20,20,20,20]
    lamda = [15,15,15,15,15,15,15]
    learning_rate = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
    alpha =  [1, 2, 5, 10]
    batch_size = [20,20,20,20,20,20]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.09,0.09,0.09,0.09,0.09,0.09] 
    algorithms = [ "Persionalized","Persionalized","Persionalized","Persionalized"]
    plot_summary_one_figure_mnist_Beta(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, alpha = alpha, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
