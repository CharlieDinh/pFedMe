import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from fedl.optimizers.fedoptimizer import MySGD, FEDLOptimizer,PersionalizedOptimizer
from fedl.users.userbase import User

class UserPersionalized(User):
    """
    User in FedAvg.dataset
    """
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate,meta_learning_rate,lamda,
                 local_epochs, optimizer):
        super().__init__(numeric_id, train_data, test_data, model[0], batch_size, learning_rate, meta_learning_rate, lamda,
                         local_epochs)

        if(model[1] == "cnn"):
            self.loss = nn.NLLLoss()
        else:
            self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        if optimizer == "SGD":
            self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate)
        if optimizer == "PersionalizedOptimizer":
            self.optimizer = PersionalizedOptimizer(self.model.parameters(), lr=self.learning_rate, lamda=self.lamda)
            #self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate)

        self.local_weight_updated = [torch.rand(self.model.fc1.weight.shape[0], self.model.fc1.weight.shape[1]),
                             torch.rand(self.model.fc1.bias.shape)]

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param = new_param.clone().requires_grad_(True)
        self.optimizer = PersionalizedOptimizer(self.model.parameters(), lr=self.learning_rate, lamda=self.lamda)
        #self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate)
        self.local_weight_updated = self.model.fc1.weight.clone()
        result = 0

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
            for batch_idx, (X, y) in enumerate(self.trainloader): # ~ t time update 
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step(self.local_weight_updated)
                #self.optimizer.step()
                loss_per_epoch += loss.item() * X.shape[0]
            loss_per_epoch /= self.train_samples
            LOSS += loss_per_epoch
            # update local weight after finding aproximate theta
            #self.local_weight_updated = self.local_weight_updated - self.lamda* self.learning_rate * (self.local_weight_updated - new_weight)
            # then update the local weight
        result = LOSS / self.local_epochs
        #print(result)
        return result

class UserFEDL(User):
    def __init__(self, numeric_id, train_data, test_data, model, batch_size, learning_rate,
                 local_epochs, optimizer, eta, lamb):
        super().__init__(numeric_id, train_data, test_data, model, batch_size, learning_rate,
                         local_epochs)

        self.lamb = lamb

        self.eta = eta

        self.server_grads = [torch.rand(self.model.fc1.weight.shape[0], self.model.fc1.weight.shape[1]),
                             torch.rand(self.model.fc1.bias.shape)]

        self.loss = self.loss_fn

        self.optimizer = FEDLOptimizer(self.model.parameters(),
                                       lr=self.learning_rate,
                                       server_grads=self.server_grads,
                                       pre_grads=self.pre_grads,
                                       eta=self.eta)
        self.optimizer.zero_grad()
        loss = self.loss(self.model(self.X_train), self.y_train)
        loss.backward()

        self.save_previous_grads()

    def loss_fn(self, input, target):
        """
        BCE with Logits Loss (reduction=mean), plus L2 regularization.
        """
        loss = F.binary_cross_entropy_with_logits(input, target)
        # Add regularization
        loss += (self.lamb / 2) * torch.sum(self.model.fc1.weight ** 2)
        return loss

    # NOT use
    def surrogate_fn(self, input, target):
        """
        J^t_n(w) = F_n(w) + <eta * \nabla \bar(F) - \nabla F_n(w^{t-1}, w>
        """
        surrogate_loss = self.loss_fn(input, target)
        surrogate_loss += torch.mm((self.eta * self.optimizer.server_grads[0] -
                                    self.optimizer.pre_grads[0]), self.model.fc1.weight.T).item()
        return surrogate_loss

    def surrogate_grads(self):
        """
        \\nabla J^t_n(w)
        """
        grads = self.model.fc1.weight.grad.data.clone()
        grads += self.eta * self.optimizer.server_grads[0]
        grads -= self.optimizer.pre_grads[0]
        return grads

    def set_parameters(self, model):

        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()
        for old_param, new_param in zip(self.server_model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()
            old_param.grad = torch.zeros_like(old_param.data)

        # Find F_n(w^{t-1})
        # self.optimizer.zero_grad()
        output = self.server_model(self.X_train)
        server_loss = self.loss(output, self.y_train)
        server_loss.backward()
        self.save_previous_grads()

    def train(self, epochs):
        LOSS = 0
        self.model.train()
        previous_surrogate_grads = self.surrogate_grads().norm(2)
        current_surrogate_grads = self.surrogate_grads().norm(2)
        i = 0
        while current_surrogate_grads > 0.8 * previous_surrogate_grads and i < self.local_epochs:
            i += 1
            loss_per_epoch = 0
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                # loss = self.surrogate_fn(output, y)
                loss.backward()
                self.optimizer.step()
                loss_per_epoch += loss.item() * X.shape[0]
            current_surrogate_grads = self.surrogate_grads().norm(2)
            loss_per_epoch /= self.train_samples
            LOSS += loss_per_epoch
        para = self.model.parameters()
        return LOSS / i

    def save_previous_grads(self):
        pre_grads = []
        for param in self.server_model.parameters():
            if param.grad is not None:
                pre_grads.append(param.grad.data.clone())
            else:
                pre_grads.append(torch.zeros_like(param.data))
        self.pre_grads = pre_grads
        self.optimizer.pre_grads = pre_grads

    def save_server_grads(self, server_grads):
        self.server_grads = server_grads.copy()
        self.optimizer.server_grads = server_grads.copy()
