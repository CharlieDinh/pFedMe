# Personalized Federated Learning with Moreau Envelopes
This repository implements all experiments in the paper the Personalized Federated Learning with Moreau Envelopes.
This repository is not only implement pFedMe but also FedAvg, Per-FedAvg, and APFL algorithms

# Software requirement
numpy

scipy

pytorch

Pillow

matplotlib

tqdm

# Dataset: We using 2 dataset: MNIST and Synthetic
- To gennerate non-idd MNIST Data: 
  - Access data/Mnist and run: "python3 generate_niid_20users.py"
  - We can change the number of user and number of labels for each user using 2 variable NUM_USERS = 20 and NUM_LABELS = 2

- To gennerate idd MNIST Data:
  - Access data/Mnist and run: "python3 generate_iid_20users.py"

- To generate niid Synthetic:
  - Access data/Synthetic and run: "python3 generate_synthetic_05_05.py". Similar to MNIST data, the sythetic data is configurable with number of users and numbers of labels for each user.



# Produce experiments and figures

There is a main file "main.py" which allows to run the experiment.

For example:
To produce the experiment for pFedMe using MNIST dataset:

python3 main.py --dataset Mnist --model Mclr_Logistic --batch_size 20 --learning_rate 0.005 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm Persionalized --numusers 5

All the train loss, testing accurancy, and training accurancy will be stored as h5py file in the folder results.
