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
from fedl.servers.serverpsnl import pFedMe
from fedl.servers.serverperavg import PerAvg
from fedl.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)
def main(dataset, algorithm, model, batch_size, learning_rate, alpha, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate):
    
    local_ep = 20
    lamda = 15
    learning_rate = 0.01
    beta =  2
    batch_size = 20
    K = 5
    personal_learning_rate = 0.01
    algorithms = "pFedMe"
    times = 5
    if(1):
        model = Mclr_Logistic(60,10), model
        for j in range(times):
            print(algorithms)
            if(algorithms == "FedAvg"):
                server = FedAvg(dataset,algorithms, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_ep, optimizer, numusers,j)
                server.train()
                server.test()
            if(algorithms == "pFedMe"):
                server = pFedMe(dataset,algorithms, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_ep, optimizer, numusers, K, personal_learning_rate,j)
                server.train()
                server.test()
            if(algorithms == "APFL"):
                server = APFL(dataset,algorithms, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_ep, optimizer, numusers,j)
                server.train()
                server.test()
            if(algorithms == "PerAvg"):
                server = PerAvg(dataset,algorithms, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_ep, optimizer, numusers,j)
                server.train()
                server.test()
    average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, alpha = beta, algorithms=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Logistic_Synthetic", choices=["Mnist", "Logistic_Synthetic"])
    parser.add_argument("--model", type=str, default="Mclr_Logistic",
                        choices=["cnn", "Mclr_Logistic", "Mclr_CrossEntropy"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--alpha", type=float, default=1, help="Mixture Weight for APFL")
    parser.add_argument("--lamda", type=float, default=3, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=600)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="Persionalized",
                        choices=["Persionalized", "PerAvg", "FedAvg", "APFL"])
    parser.add_argument("--numusers", type=float, default=10, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=10, help="Optimization steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Personal learning rate")
    args = parser.parse_args()
    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("local learing rate       : {}".format(args.learning_rate))
    print("meta learing rate       : {}".format(args.alpha))
    print("number user per round       : {}".format(args.numusers))
    print("K_g       : {}".format(args.num_global_iters))
    print("K_l       : {}".format(args.local_epochs))
    print("=" * 80)
    main(
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        alpha = args.alpha, 
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate
    )