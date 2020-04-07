#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from fedl.servers.serveravg import FedAvg
from fedl.servers.serverfedl import FEDL
from fedl.servers.serverpsnl import Persionalized
from fedl.trainmodel.models import Mclr_Logistic,Net
from utils.plot_utils import plot_summary_one_figure

def main(dataset, algorithm, model, batch_size, learning_rate, meta_learning_rate, lamda, num_glob_iters,
         local_epochs, optimizer,numusers):
    if(1):
        if(model == "Mclr_Synthetic"):
            model = Mclr_Logistic(40,2), model
        else:
            model = Mclr_Logistic(), model
            
        if(model == "cnn"):
            model = Net(), model

        if(algorithm == "FedAvg"):
            server = FedAvg(dataset,algorithm, model, batch_size, learning_rate, meta_learning_rate, lamda, num_glob_iters, local_epochs, optimizer, numusers)
    
        if(algorithm == "Persionalized"):
            server = Persionalized(dataset,algorithm, model, batch_size, learning_rate, meta_learning_rate, lamda, num_glob_iters, local_epochs, optimizer, numusers)

        server.train()
        server.test()

    # plot the result:
    plot_summary_one_figure(num_users=numusers, loc_ep1=[local_epochs], Numb_Glob_Iters=num_glob_iters, lamb=[lamda],
                               learning_rate=[learning_rate], meta_learning_rate = [meta_learning_rate], algorithms_list=[algorithm], batch_size=[batch_size], dataset=dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist",
                        choices=["Mnist", "Logistic_Synthetic"])
    parser.add_argument("--model", type=str, default="Mclr_Mnist", choices=["cnn", "Mclr_Mnist", "Mclr_Synthetic"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--meta_learning_rate", type=float, default=1, help="Meta learning rate for global round")
    parser.add_argument("--lamda", type=float, default = 3, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="Persionalized")
    parser.add_argument("--numusers", type=float, default=10, help="Number of Users per round") 
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("local learing rate       : {}".format(args.learning_rate))
    print("meta learing rate       : {}".format(args.meta_learning_rate))
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
        meta_learning_rate = args.meta_learning_rate, 
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers
    )
