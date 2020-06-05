# Personalized Federated Learning with Moreau Envelopes
This repository implements all experiments in the paper the <mark>Personalized Federated Learning with Moreau Envelopes<mark>.
  
Authors: Anonymous

This repository not only implements pFedMe but also FedAvg, Per-FedAvg, and APFL algorithms. \
(Federated Learning using Pytorch)

# Software requirement:
- numpy

- scipy

- pytorch

- Pillow

- matplotlib

# Dataset: We use 2 datasets: MNIST and Synthetic
- To generate non-idd MNIST Data: 
  - Access data/Mnist and run: "python3 generate_niid_20users.py"
  - We can change the number of user and number of labels for each user using 2 variable NUM_USERS = 20 and NUM_LABELS = 2

- To generate idd MNIST Data:
  - Access data/Mnist and run: "python3 generate_iid_20users.py"

- To generate niid Synthetic:
  - Access data/Synthetic and run: "python3 generate_synthetic_05_05.py". Similar to MNIST data, the Synthetic data is configurable with number of users and the numbers of labels for each user.

- The datasets also are available to download at: https://drive.google.com/drive/folders/1vTrQzE3Ww-Oc0c3apFt8tkRvw5eW07eL

# Produce experiments and figures

- There is a main file "main.py" which allows running the experiment.
## Using same parameters
- To produce the comparison experiments for pFedMe using MNIST dataset:
![MNIST](https://user-images.githubusercontent.com/44039773/83833168-a9f59680-a72e-11ea-9787-88cc150fdb53.png)

  - Strongly Convex Case:
    <pre><code>
    python3 main.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.005 --personal_learning_rate 0.1 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 5 --times 10
    python3 main.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.005 --num_global_iters 800 --local_epochs 20 --algorithm FedAvg --numusers 5  --times 10
    python3 main.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.005 --beta 0.001  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 5  --times 10
    </code></pre>
  
  - Non-Convex case: 
    <pre><code>
    python3 main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.005 --personal_learning_rate 0.09 --beta 1 --lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 5 -- times 10
    python3 main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.005 --num_global_iters 800 --local_epochs 20 --algorithm FedAvg --numusers 5 -- times 10
    python3 main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.005 --beta 0.001  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 5 -- times 10
    </code></pre>

- It is noted that each algorithm should be run at least 10 times and then the results are averaged.

- All the train loss, testing accuracy, and training accuracy will be stored as h5py file in the folder "results". It is noted that we store the data for persionalized model and global in 2 separate files following format: DATASET_pFedMe_p_x_x_xu_xb_x_avg.h5 and DATASET_pFedMe_x_x_xu_xb_x_avg.h5 respectively. 

- In order to plot the figure, set parameters in file main_plot.py similar to parameters run from previous experiments.
  For example. To plot the 3 experiments above, in the main_plot.py set:
   <pre><code>
    numusers = 5
    num_glob_iters = 800
    dataset = "Mnist"
    local_ep = [20,20,20,20]
    lamda = [15,15,15,15]
    learning_rate = [0.005, 0.005, 0.005, 0.005]
    beta =  [1.0, 1.0, 0.001, 1.0]
    batch_size = [20,20,20,20]
    K = [5,5,5,5]
    personal_learning_rate = [0.1,0.1,0.1,0.1]
    algorithms = [ "pFedMe_p","pFedMe","PerAvg_p","FedAvg"]
    plot_summary_one_figure_mnist_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                               learning_rate=learning_rate, alpha = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
    </code></pre>

- To produce the comparision experiment for pFedMe using Synthetic dataset:
![SYNTHETIC](https://user-images.githubusercontent.com/44039773/83833171-ac57f080-a72e-11ea-90c7-c8480d275fff.png)

  - Strongly Convex Case:
    <pre><code>
    python3 main.py --dataset Synthetic --model mclr --batch_size 20 --learning_rate 0.005 --personal_learning_rate 0.01 --beta 1 --lamda 20 --num_global_iters 600 --local_epochs 20 --algorithm pFedMe --numusers 10 -- times 10
    python3 main.py --dataset Synthetic --model mclr --batch_size 20 --learning_rate 0.005 --num_global_iters 600 --local_epochs 20 --algorithm FedAvg --numusers 10 -- times 10
    python3 main.py --dataset Synthetic --model mclr --batch_size 20 --learning_rate 0.005 --beta 0.001  --num_global_iters 600 --local_epochs 20 --algorithm PerAvg --numusers 10 -- times 10
    </code></pre>

  - Non-Convex case: 
    <pre><code>
    python3 main.py --dataset Synthetic --model dnn --batch_size 20 --learning_rate 0.005 --personal_learning_rate 0.09 --beta 1 --lamda 20 --num_global_iters 600 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 10
    python3 main.py --dataset Synthetic --model dnn --batch_size 20 --learning_rate 0.005 --num_global_iters 600 --local_epochs 20 --algorithm FedAvg --numusers 10 --times 10
    python3 main.py --dataset Synthetic --model dnn --batch_size 20 --learning_rate 0.005 --beta 0.001  --num_global_iters 600 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 10
    </code></pre>

## Fine-tuned Parameter:
To produce results in the table of fine-tune parameter:
- MNIST:
  - Convex:
      <pre><code>
    python3 main.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.1--beta 2--lamda 15 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 5 --times 10
    python3 main.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.02 --num_global_iters 800 --local_epochs 20 --algorithm FedAvg --numusers 5 --times 10
    python3 main.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.03 --beta 0.003  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 5 --times 10
    </code></pre>
  - NonConvex:
    <pre><code>
    python3 main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.005 --personal_learning_rate 0.05 --beta 1 --lamda 30 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 5 --times 10
    python3 main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.02 --num_global_iters 800 --local_epochs 20 --algorithm FedAvg --numusers 5 --times 10
    python3 main.py --dataset Mnist --model dnn --batch_size 20 --learning_rate 0.02 --beta 0.001  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 5 --times 10
    </code></pre>
- Sythetic:
  - Convex:
          <pre><code>
    python3 main.py --dataset Synthetic --model mclr --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --lamda 20 --num_global_iters 600 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 10
    python3 main.py --dataset Synthetic --model mclr --batch_size 20 --learning_rate 0.02 --num_global_iters 600 --local_epochs 20 --algorithm FedAvg --numusers 10 --times 10
    python3 main.py --dataset Synthetic --model mclr --batch_size 20 --learning_rate 0.02 --beta 0.002  --num_global_iters 600 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 10
    </code></pre>
    
  - NonConvex:
      <pre><code>
    python3 main.py --dataset Synthetic --model dnn --batch_size 20 --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --lamda 30 --num_global_iters 600 --local_epochs 20 --algorithm pFedMe --numusers 10 --times 10
    python3 main.py --dataset Synthetic --model dnn --batch_size 20 --learning_rate 0.03 --num_global_iters 600 --local_epochs 20 --algorithm FedAvg --numusers 10 --times 10
    python3 main.py --dataset Synthetic --model dnn --batch_size 20 --learning_rate 0.01 --beta 0.001  --num_global_iters 600 --local_epochs 20 --algorithm PerAvg --numusers 10 --times 10
    </code></pre>
