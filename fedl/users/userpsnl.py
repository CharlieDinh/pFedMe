import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from fedl.optimizers.fedoptimizer import PersionalizedOptimizer
from fedl.users.userbase import User
import copy

class UserPersionalized(User):
    """
    User in FedAvg.dataset
    """
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate,alpha,lamda,
                 local_epochs, optimizer):
        super().__init__(numeric_id, train_data, test_data, model[0], batch_size, learning_rate, alpha, lamda,
                         local_epochs)

        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

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
            #for _ , (X, y) in enumerate(self.trainloader): # ~ t time update
            try:
                # Samples a new batch for persionalizing
                (X, y) = next(self.iter_trainloader)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                self.iter_trainloader = iter(self.trainloader)
                (X, y) = next(self.iter_trainloader)

            K = 10 # K is number of personalized steps
            for i in range(K):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.persionalized_model_bar, _ = self.optimizer.step(self.local_model)

            # update local weight after finding aproximate theta
            for new_param, localweight in zip(self.persionalized_model_bar, self.local_model):
                localweight.data = localweight.data - self.lamda* self.learning_rate * (localweight.data - new_param.data)

        #update local model as local_weight_upated
        #self.clone_model_paramenter(self.local_weight_updated, self.local_model)
        self.update_parameters(self.local_model)

        return LOSS