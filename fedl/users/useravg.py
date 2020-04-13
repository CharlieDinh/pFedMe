import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from fedl.users.userbase import User

class UserAVG(User):
    """
    User in FedAvg.dataset
    """
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate, alpha, lamda,
                 local_epochs, optimizer):
        super().__init__(numeric_id, train_data, test_data, model[0], batch_size, learning_rate, alpha, lamda,
                         local_epochs)

        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

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
            try:
                # Samples a new batch for persionalizing
                (X, y) = next(self.iter_trainloader)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                self.iter_trainloader = iter(self.trainloader)
                (X, y) = next(self.iter_trainloader)
                
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
        return LOSS

