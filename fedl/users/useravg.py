import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from fedl.optimizers.fedoptimizer import MySGD, FEDLOptimizer
from fedl.users.userbase import User

class UserAVG(User):
    """
    User in FedAvg.dataset
    """
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate, meta_learning_rate, lamda,
                 local_epochs, optimizer):
        super().__init__(numeric_id, train_data, test_data, model[0], batch_size, learning_rate, meta_learning_rate, lamda,
                         local_epochs)
        if(model[1] == "cnn" or model[1] == "Mclr_Mnist"):
            self.loss = nn.NLLLoss()
        else:
            self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        #if optimizer == "SGD":
        self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate)

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param = new_param.clone().requires_grad_(True)
        self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate)

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
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            loss_per_epoch = 0
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
                loss_per_epoch += loss.item() * X.shape[0]
            loss_per_epoch /= self.train_samples
            LOSS += loss_per_epoch
        result = LOSS / self.local_epochs
        #print(result)
        return result

