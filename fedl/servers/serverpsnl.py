import torch
import os

from fedl.users.userpsnl import UserPersionalized
from fedl.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

class Persionalized(Server):
    def __init__(self, dataset,algorithm, model, batch_size, learning_rate, alpha, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users):
        super().__init__(dataset,algorithm, model[0], batch_size, learning_rate, alpha, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users)

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserPersionalized(id, train, test, model, batch_size, learning_rate, alpha, lamda, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating Persionalized server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.evaluate()
            #self.train_evaluate()

            # do update for all users not only selected users
            for user in self.users:
                user.train(self.local_epochs) #* user.train_samples
            
            # choose several users to send back upated model to server
            # self.personalized_evaluate()

            self.selected_users = self.select_users(glob_iter,self.num_users)

            # Evaluate gloal model on user for each interation
            print("Evaluate persionalized model")
            print("")
            self.evaluate_personalized_model()

            self.persionalized_aggregate_parameters()

            #loss_ /= self.total_train_samples
            #loss.append(loss_)
            #print(loss_)

        #print(loss)
        self.save_results()
        self.save_model()
    
  
