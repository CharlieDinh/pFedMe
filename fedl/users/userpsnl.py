import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from fedl.optimizers.fedoptimizer import MySGD, FEDLOptimizer,PersionalizedOptimizer
from fedl.users.userbase import User
import copy

class UserPersionalized(User):
    """
    User in FedAvg.dataset
    """
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate,meta_learning_rate,lamda,
                 local_epochs, optimizer):
        super().__init__(numeric_id, train_data, test_data, model[0], batch_size, learning_rate, meta_learning_rate, lamda,
                         local_epochs)

        if(model[1] == "cnn" or model[1] == "Mclr_Mnist"):
            self.loss = nn.NLLLoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        #if optimizer == "SGD":
        #    self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate)
        #if optimizer == "PersionalizedOptimizer":
        
        self.optimizer = PersionalizedOptimizer(self.model.parameters(), lr=self.learning_rate, lamda=self.lamda)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):  # local update 
            self.model.train()
            loss_per_epoch = 0
            #dataloader_iterator = iter(self.trainloader)
            for _ , (X, y) in enumerate(self.trainloader): # ~ t time update
            #for i in range(4):
                #(X, y) = next(dataloader_iterator)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.persionalized_model, _ = self.optimizer.step(self.local_weight_updated)
                loss_per_epoch += loss.item() * X.shape[0]
            loss_per_epoch /= self.train_samples
            LOSS += loss_per_epoch
            
            # update local weight after finding aproximate theta
            for new_param, localweight in zip(self.persionalized_model, self.local_weight_updated):
                localweight.data = localweight.data - self.lamda* self.learning_rate * (localweight.data - new_param.data)
        
        # evaluate personal model before making argeation at the server size 
        #test_acc, _ = self.test()
        #print("check accurancy of peronal model ", test_acc)
        self.update_parameters(self.local_weight_updated)

        result = LOSS / self.local_epochs
        #print(result)
        return result
    
    def train_evaluate(self, epochs):
        LOSS = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):  # local update
            self.model.train()
            loss_per_epoch = 0
            #dataloader_iterator = iter(self.testloader)
            for _, (X, y) in enumerate(self.testloader):  # ~ t time update
            #for i in range(4):
            #    (X, y) = next(dataloader_iterator)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                new_params, _ = self.optimizer.step(self.local_weight_updated)
                loss_per_epoch += loss.item() * X.shape[0]

            # update local weight after finding aproximate theta
            for new_param, localweight in zip(new_params, self.local_weight_updated):
                localweight.data = localweight.data - self.lamda * \
                    self.learning_rate * (localweight.data - new_param.data)

        self.update_parameters(self.local_weight_updated)
