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
from utils.plot_utils import plot_summary_one_figure
import torch
torch.manual_seed(0)

def main(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate):
    if(1):
        # Generate model
        if(model == "mclr"):
            if(dataset == "Mnist"):
                model = Mclr_Logistic(), model
            else:
                model = Mclr_Logistic(60,10), model
            
        if(model == "cnn"):
            model = Net(), model
        
        if(model == "dnn"):
            if(dataset == "Mnist"):
                model = DNN(), model
            else: 
                model = DNN(60,20,10), model
        
        # select algorithm
        if(algorithm == "FedAvg"):
            server = FedAvg(dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers)
    
        if(algorithm == "Persionalized"):
            server = Persionalized(dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers,K,personal_learning_rate )

        if(algorithm == "APFL"):
            server = APFL(dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers)

        if(algorithm == "PerAvg"):
            server = PerAvg(dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers)

        server.train()
        server.test()

    # plot the result:
    plot_summary_one_figure(num_users=numusers, loc_ep1=[local_epochs], Numb_Glob_Iters=num_glob_iters, lamb=[lamda],
                               learning_rate=[learning_rate], alpha = [beta], algorithms_list=[algorithm], batch_size=[batch_size], dataset=dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist", choices=["Mnist", "Synthetic"])
    parser.add_argument("--model", type=str, default="dnn", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1, help="Average moving parameter for pFedMe, or Learning rate of Per-FedAvg, or Mixmodel of APFL")
    parser.add_argument("--lamda", type=float, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=800)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="Persionalized",
                        choices=["Persionalized", "PerAvg", "FedAvg", "APFL"])
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Optimization steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.1, help="Personal learning rate")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("local learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
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
        beta = args.beta, 
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate
        )
