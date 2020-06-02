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
from fedl.trainmodel.models import Mclr_Logistic, Net, Mclr_CrossEntropy, DNN
from utils.plot_utils import *
import torch
torch.manual_seed(0)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def main(dataset, algorithm, model, batch_size, learning_rate, alpha, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate):
    
    local_ep = [20]
    lamda = [15]
    learning_rate = [0.005]
    beta =  [0.001]
    batch_size = [20]
    K = [5]
    personal_learning_rate = [0.1] 
    algorithms = ["pFedMe"]
    times = 5
    if(1):
        model = Mclr_Logistic(), model
        
        for i in range(len(algorithms)):
            for j in range(times):
                print(algorithms[i])
                if(algorithms[i] == "FedAvg"):
                    server = FedAvg(dataset,algorithms[i], model, batch_size[i], learning_rate[i], beta[i], lamda[i], num_glob_iters, local_ep[i], optimizer, numusers,j)
                    server.train()
                    server.test()
                if(algorithms[i] == "pFedMe"):
                    server = pFedMe(dataset,algorithms[i], model, batch_size[i], learning_rate[i], beta[i], lamda[i], num_glob_iters, local_ep[i], optimizer, numusers, K[i], personal_learning_rate[i],j)
                    server.train()
                    server.test()
                if(algorithms[i] == "APFL"):
                    server = APFL(dataset,algorithms[i], model, batch_size[i], learning_rate[i], beta[i], lamda[i], num_glob_iters, local_ep[i], optimizer, numusers,j)
                    server.train()
                    server.test()
                if(algorithms[i] == "PerAvg"):
                    server = PerAvg(dataset,algorithms[i], model, batch_size[i], learning_rate[i], beta[i], lamda[i], num_glob_iters, local_ep[i], optimizer, numusers,j)
                    server.train()
                    server.test()
                    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist", choices=["Mnist", "Logistic_Synthetic"])
    parser.add_argument("--model", type=str, default="dnn",
                        choices=["dnn", "Mclr_Logistic", "Mclr_CrossEntropy", "cnn"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--alpha", type=float, default=1, help="Mixture Weight for APFL")
    parser.add_argument("--lamda", type=float, default=3, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="Persionalized",
                        choices=["Persionalized", "PerAvg", "FedAvg", "APFL"])
    parser.add_argument("--numusers", type=float, default=5, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Optimization steps")
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
