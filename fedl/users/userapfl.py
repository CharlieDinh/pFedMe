import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from fedl.users.userbase import User
from fedl.optimizers.fedoptimizer import APFLOptimizer
import copy

class UserAPFL(User):
    """
    User in FedAvg.dataset
    """
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate,alpha,lamda,
                 local_epochs, optimizer, total_users , num_users):
        super().__init__(numeric_id, train_data, test_data, model[0], batch_size, learning_rate, alpha, lamda,
                         local_epochs)
        self.total_users = total_users
        self.num_users = num_users
        
        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = APFLOptimizer(self.model.parameters(), lr=self.learning_rate, lamda=self.lamda)

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

            try:
                # Samples a new batch for persionalizing
                (X, y) = next(self.iter_trainloader)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                self.iter_trainloader = iter(self.trainloader)
                (X, y) = next(self.iter_trainloader)
            
            # caculate local model 
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
            self.local_model = list(self.model.parameters()).copy()

            # caculate persionalized model
            self.update_parameters(self.persionalized_model)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step(self.alpha,self.total_users/self.num_users)
            self.persionalized_model = list(self.model.parameters()).copy() 

            # caculate persionalized bar model
            for persionalized_bar, persionalized, local in zip(self.persionalized_model_bar, self.persionalized_model, self.local_model):
                persionalized_bar = self.alpha * persionalized + (1 - self.alpha )* local
        
        #self.update_parameters(self.local_weight_updated)

        return LOSS