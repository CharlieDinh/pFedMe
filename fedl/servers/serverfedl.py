import torch
import os

from fedl.users.userfedl import UserFEDL
from fedl.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data

class FEDL(Server):
    def __init__(self, dataset, model, batch_size, learning_rate, num_glob_iters,
                 local_epochs, optimizer, num_users=100, eta=0.25, lamb=0):
        super().__init__(dataset, model, batch_size, learning_rate, num_glob_iters,
                         local_epochs, optimizer, num_users)

        # Hyper-learning rate
        self.eta = eta

        # Regularization rate
        self.lamb = lamb

        #selected_clients = self.select_clients(10, num_clients=20)
        for i in range(num_users):
            user = UserFEDL(i, dataset, model, batch_size, learning_rate,
                            local_epochs, optimizer, eta, lamb)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        self.send_parameters()
        self.train_loss = torch.zeros(self.num_glob_iters, dtype=torch.float32)
        print("Finished creating server.")

    def send_grads(self, glob_iter=0):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.save_server_grads(grads)

    def train(self):
        for glob_iter in range(self.num_glob_iters):

            # 1: Find \nabla \bar{F} and send it to users
            self.aggregate_grads()
            self.send_grads()

            # 2: Local problem-solving (find w_t^n)
            loss = 0
            for user in self.users:
                loss += user.train(self.local_epochs) * user.train_samples / self.total_train_samples
            print(loss)
            self.train_loss[glob_iter] = loss

            # 3: Find w^t and send it to users (to find \nabla \F_n(w^t))
            self.aggregate_parameters()
            self.send_parameters()

        print(self.train_loss)
        import matplotlib.pyplot as plt
        plt.plot(range(self.num_glob_iters), self.train_loss)
        plt.savefig("josh.png")
        self.save_model()
