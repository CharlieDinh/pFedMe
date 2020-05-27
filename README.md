# Personalized Federated Learning with Moreau Envelopes
This repository implements all experiments in the paper the Personalized Federated Learning with Moreau Envelopes. 

This repository not only implements pFedMe but also FedAvg, Per-FedAvg, and APFL algorithms.

# Software requirement
numpy

scipy

pytorch

Pillow

matplotlib

# Dataset: We use 2 datasets: MNIST and Synthetic
- To generate non-idd MNIST Data: 
  - Access data/Mnist and run: "python3 generate_niid_20users.py"
  - We can change the number of user and number of labels for each user using 2 variable NUM_USERS = 20 and NUM_LABELS = 2

- To generate idd MNIST Data:
  - Access data/Mnist and run: "python3 generate_iid_20users.py"

- To generate niid Synthetic:
  - Access data/Synthetic and run: "python3 generate_synthetic_05_05.py". Similar to MNIST data, the Synthetic data is configurable with number of users and numbers of labels for each user.

- The datasets also are available to download at: https://drive.google.com/drive/folders/1vTrQzE3Ww-Oc0c3apFt8tkRvw5eW07eL

# Produce experiments and figures

- There is a main file "main.py" which allows to run the experiment.

- To produce the comparision experiment for pFedMe using MNIST dataset:
![MNIST](https://user-images.githubusercontent.com/44039773/82981428-6795d080-a02f-11ea-9e0b-9da210354975.png)

- Strongly Convex Case:
<pre><code>
python3 main.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.005 --personal_learning_rate 0.1 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm Persionalized --numusers 5
python3 main.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.005 --num_global_iters 800 --local_epochs 20 --algorithm FedAvg --numusers 5
python3 main.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.005 --beta 0.001  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 5
</code></pre>
  
- Non-Convex case: 
<pre><code>
python3 main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.005 --personal_learning_rate 0.09 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm Persionalized --numusers 5
python3 main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.005 --num_global_iters 800 --local_epochs 20 --algorithm FedAvg --numusers 5
python3 main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.005 --beta 0.001  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 5
</code></pre>

- All the train loss, testing accuracy, and training accuracy will be stored as h5py file in the folder "results". It is noted that we store the data for persionalized model of pFedMe in xxPersionalized_pxx.h5 file while the global model is stored in xxPersionalizedxx.h5

- In order to plot the figure, set parameters in file main_plot.py similar to parameters run from previous experiments.
   For example. To plot the 3 experiments above, in the main_plot.py set:
   <pre><code>
    algorithms = ["Persionalized","Persionalized","PerAvg","FedAvg"]
    local_ep = [20,20,2020]
    lamda = [15,15,15,15]
    learning_rate = [0.005,0.005,0.005,0.005]
    alpha =  [1,1,1,0.001]
    batch_size = [20,20,20,20]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.1,0.1,0.1,0.1] 
    plot_summary_one_figure_mnist_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda, learning_rate=learning_rate, alpha = alpha, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
    </code></pre>

- To produce the comparision experiment for pFedMe using Synthetic dataset:
![SYNTHETIC](https://user-images.githubusercontent.com/44039773/82981423-65cc0d00-a02f-11ea-92e4-b4fc09989d69.png)

- Strongly Convex Case:
<pre><code>
python3 main.py --dataset Synthetic --model mclr --batch_size 20 --learning_rate 0.005 --personal_learning_rate 0.1 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm Persionalized --numusers 5
python3 main.py --dataset Synthetic --model mclr --batch_size 20 --learning_rate 0.005 --num_global_iters 800 --local_epochs 20 --algorithm FedAvg --numusers 5
python3 main.py --dataset Synthetic --model mclr --batch_size 20 --learning_rate 0.005 --beta 0.001  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 5
</code></pre>

- Non-Convex case: 
<pre><code>
python3 main.py --dataset Synthetic --model dnn --batch_size 20 --learning_rate 0.005 --personal_learning_rate 0.09 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm Persionalized --numusers 5
python3 main.py --dataset Synthetic --model dnn --batch_size 20 --learning_rate 0.005 --num_global_iters 800 --local_epochs 20 --algorithm FedAvg --numusers 5
python3 main.py --dataset Synthetic --model dnn --batch_size 20 --learning_rate 0.005 --beta 0.001  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 5
</code></pre>
