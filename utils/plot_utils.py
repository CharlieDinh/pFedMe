import matplotlib.pyplot as plt
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import os

def simple_read_data(alg):
    print(alg)
    hf = h5py.File("./results/"+'{}.h5'.format(alg), 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    return rs_train_acc, rs_train_loss, rs_glob_acc

def get_training_data_value(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[],alpha=[],algorithms_list=[], batch_size=[], dataset="", k= [] , personal_learning_rate = []):
    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        string_learning_rate = str(learning_rate[i])  
        string_learning_rate = string_learning_rate + "_" +str(alpha[i]) + "_" +str(lamb[i])
        if(algorithms_list[i] == "Persionalized" or algorithms_list[i] == "Persionalized_p"):
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b" + "_" +str(loc_ep1[i]) + "_"+ str(k[i])  + "_"+ str(personal_learning_rate[i])
        else:
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b"  "_" +str(loc_ep1[i])

        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(dataset +"_"+ algorithms_list[i]))[:, :Numb_Glob_Iters]
        algs_lbl[i] = algs_lbl[i]
    return glob_acc, train_acc, train_loss


def get_data_label_style(input_data = [], linestyles= [], algs_lbl = [], lamb = [], loc_ep1 = 0, batch_size =0):
    data, lstyles, labels = [], [], []
    for i in range(len(algs_lbl)):
        data.append(input_data[i, ::])
        lstyles.append(linestyles[i])
        labels.append(algs_lbl[i]+str(lamb[i])+"_" +
                      str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b")

    return data, lstyles, labels
  
def plot_summary_one_figure(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], alpha=[], algorithms_list=[], batch_size=0, dataset = "", k = [], personal_learning_rate = []):
    Numb_Algs = len(algorithms_list)
    dataset = dataset
    glob_acc, train_acc, train_loss = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, alpha, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    plt.figure(1)
    MIN = train_loss.min() - 0.001
    start = 0
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for i in range(Numb_Algs):
        plt.plot(train_acc[i, 1:], linestyle=linestyles[i], label=algorithms_list[i] + str(lamb[i])+ "_"+str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b")
    plt.legend(loc='lower right')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Global rounds ' + '$K_g$')
    plt.title(dataset.upper())
    #plt.ylim([0.88, train_acc.max() + 0.01])
    plt.savefig(dataset.upper() + str(loc_ep1[0]) + 'train_acc.png')
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')
    plt.figure(2)
    for i in range(Numb_Algs):
        plt.plot(train_loss[i, start:], linestyle=linestyles[i], label=algorithms_list[i] + str(lamb[i]) +
                 "_"+str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b")
        #plt.plot(train_loss1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='upper right')
    #plt.ylim([MIN, MIN+ 0.3])
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.title(dataset.upper())
    plt.ylim([train_loss.min(), 0.6])
    plt.savefig(dataset.upper() + str(loc_ep1[0]) + 'train_loss.png')
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf')
    plt.figure(3)
    for i in range(Numb_Algs):
        plt.plot(glob_acc[i, start:], linestyle=linestyles[i],
                 label=algorithms_list[i]+str(lamb[i])+"_"+str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b")
        #plt.plot(glob_acc1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='lower right')
    #plt.ylim([0.6, glob_acc.max()])
    #plt.ylim([0.88,  glob_acc.max() + 0.01])
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    plt.savefig(dataset.upper() + str(loc_ep1[0]) + 'glob_acc.png')
    #plt.savefig(dataset + str(loc_ep1[1]) + 'glob_acc.pdf')

def get_max_value_index(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[], batch_size=0, dataset=""):
    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value(
        num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, algorithms_list, batch_size, dataset)
    for i in range(Numb_Algs):
        print("Algorithm: ", algorithms_list[i], "Max testing Accurancy: ", glob_acc[i].max(
        ), "Index: ", np.argmax(glob_acc[i]), "local update:", loc_ep1[i])

def get_label_name(name):
    if name.startswith("Persionalized"):
        if name.startswith("Persionalized_p"):
            return "PFedME (Personalized Model)"
        else:
            return "PFedME (Global Model)"
    if name.startswith("PerAvg"):
        return "Per-FedAvg"
    if name.startswith("FedAvg"):
        return "FedAvg"
    if name.startswith("APFL"):
        return "APFL"


def plot_summary_one_figure_synthetic_R(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, alpha, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    glob_acc, train_acc, train_loss = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,alpha, algorithms_list, batch_size, dataset, k,personal_learning_rate)
    one_alg = all([alg == algorithms_list[0] for alg in algorithms_list])
    linestyles = ['-','-','-','-','-','-','-']
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'c', 'darkorange', 'tab:brown', 'w']
    markers = ["o","v","s","*","x","P"]
    plt.figure(1)
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], label=label + ": "
                 r'$R = $' +str(loc_ep1[i]), linewidth = 1, color=colors[i], marker = markers[i],markevery=0.05, markersize=4)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([train_loss.min() - 0.01,  2])
    #plt.ylim([0.5,  1.8])
    plt.savefig(dataset.upper() + "Non_Convex_Syn_fixR.pdf", bbox_inches="tight")

    plt.figure(2)
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], label=label + ": "
                 r'$R = $' +str(loc_ep1[i]), linewidth = 1, color=colors[i], marker = markers[i],markevery=0.05, markersize=4)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.6,  0.86])
    #plt.ylim([0.89,  0.945])
    plt.savefig(dataset.upper() + "Non_Convex_Syn_fixR_test.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Convex_Syn_fixR.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_synthetic_K(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, alpha, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    glob_acc, train_acc, train_loss = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,alpha, algorithms_list, batch_size, dataset, k,personal_learning_rate)
    one_alg = all([alg == algorithms_list[0] for alg in algorithms_list])
    linestyles = ['-', '--', '-.','-', '--', '-.']
    linestyles = ['-','-','-','-','-','-','-']
    print(lamb)
    colors = ['tab:blue', 'tab:green','darkorange', 'r', 'c', 'tab:brown', 'w']
    markers = ["o","v","s","*","x","P"]
    plt.figure(1)
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + ": "
                 r'$K = $' +str(k[i]), linewidth = 1, color=colors[i], marker = markers[i],markevery=0.05, markersize=4)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss', size=14)
    plt.xlabel('Global rounds', size=14)
    #plt.ylim([0.5,  1.8])
    plt.ylim([train_loss.min() - 0.01,  2])
    #plt.savefig(dataset.upper() + "Non_Convex_Syn_fixK.pdf", bbox_inches="tight")
    plt.savefig(dataset.upper() + "Convex_Syn_fixK.pdf", bbox_inches="tight")
    plt.figure(2)
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + ": "
                 r'$K = $' +str(k[i]), linewidth = 1, color=colors[i], marker = markers[i],markevery=0.05, markersize=4)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.6,  0.86])
    plt.savefig(dataset.upper() + "Convex_Syn_fixK_test.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_synthetic_L(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, alpha, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    glob_acc, train_acc, train_loss = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,alpha, algorithms_list, batch_size, dataset, k,personal_learning_rate)
    one_alg = all([alg == algorithms_list[0] for alg in algorithms_list])
    linestyles = ['-', '--', '-.',':','-', '--', '-.']
    linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'c', 'darkorange', 'tab:brown', 'm']
    plt.figure(1)
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + ": "
                 + r'$\lambda = $'+ str(lamb[i]), linewidth = 1, color=colors[i],marker = markers[i],markevery=0.05, markersize=4)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.5,  1.8])
    #plt.ylim([train_loss.min() - 0.01,  2])
    #plt.savefig(dataset.upper() + "Non_Convex_Syn_fixL.pdf", bbox_inches="tight")
    plt.savefig(dataset.upper() + "Convex_Syn_fixL.pdf", bbox_inches="tight")
    plt.figure(2)
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + ": "
                 + r'$\lambda = $'+ str(lamb[i]), linewidth = 1, color=colors[i],marker = markers[i],markevery=0.05, markersize=4)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.6,  0.86])
    plt.savefig(dataset.upper() + "Convex_Syn_fixL_test.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_synthetic_D(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, alpha, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    glob_acc, train_acc, train_loss = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,alpha, algorithms_list, batch_size, dataset, k,personal_learning_rate)
    one_alg = all([alg == algorithms_list[0] for alg in algorithms_list])
    linestyles = ['-', '--', '-.',':','-', '--', '-.']
    linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'c', 'darkorange', 'tab:brown', 'm']
    plt.figure(1)
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + ": "
                 + r'$\lambda = $'+ str(lamb[i]), linewidth = 1, color=colors[i],marker = markers[i],markevery=0.05, markersize=4)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.5,  1.8])
    #plt.ylim([train_loss.min() - 0.01,  2])
    #plt.savefig(dataset.upper() + "Non_Convex_Syn_fixL.pdf", bbox_inches="tight")
    plt.savefig(dataset.upper() + "Convex_Syn_fixL.pdf", bbox_inches="tight")
    plt.figure(2)
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + ": "
                 + r'$\lambda = $'+ str(lamb[i]), linewidth = 1, color=colors[i],marker = markers[i],markevery=0.05, markersize=4)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.6,  0.86])
    plt.savefig(dataset.upper() + "Convex_Syn_fixL_test.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_synthetic_Compare(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, alpha, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    glob_acc, train_acc, train_loss = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,alpha, algorithms_list, batch_size, dataset, k,personal_learning_rate)
    one_alg = all([alg == algorithms_list[0] for alg in algorithms_list])
    linestyles = ['-', '--', '-.',':','-', '--', '-.']
    linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'c', 'darkorange', 'tab:brown', 'm']
    plt.figure(1)
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + ": "
                 + r'$\lambda = $'+ str(lamb[i]), linewidth = 1, color=colors[i],marker = markers[i],markevery=0.05, markersize=4)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.5,  1.8])
    #plt.ylim([train_loss.min() - 0.01,  2])
    #plt.savefig(dataset.upper() + "Non_Convex_Syn_fixL.pdf", bbox_inches="tight")
    plt.savefig(dataset.upper() + "Convex_Syn_fixL.pdf", bbox_inches="tight")
    plt.figure(2)
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + ": "
                 + r'$\lambda = $'+ str(lamb[i]), linewidth = 1, color=colors[i],marker = markers[i],markevery=0.05, markersize=4)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.6,  0.86])
    plt.savefig(dataset.upper() + "Convex_Syn_fixL_test.pdf", bbox_inches="tight")
    plt.close()


def plot_summary_one_figure_mnist_Compare(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, alpha, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    glob_acc, train_acc, train_loss = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,alpha, algorithms_list, batch_size, dataset, k,personal_learning_rate)
    one_alg = all([alg == algorithms_list[0] for alg in algorithms_list])
    linestyles = ['-', '--', '-.','-', '--', '-.']
    linestyles = ['-','-','-','-','-','-','-']
    #linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1)
    plt.grid(True)
    # training loss
    marks = []
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label, linewidth = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=4)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.18,  0.5])
    plt.savefig(dataset.upper() + "Convex_Mnist_train.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train.pdf", bbox_inches="tight")
    plt.figure(2)
    plt.grid(True)
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label, linewidth = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=4)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.9,  0.947])
    #plt.ylim([0.89,  0.945])
    plt.savefig(dataset.upper() + "Convex_Mnist_test.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_mnist_K(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, alpha, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    glob_acc, train_acc, train_loss = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,alpha, algorithms_list, batch_size, dataset, k,personal_learning_rate)
    linestyles = ['-', '--', '-.','-', '--', '-.']
    linestyles = ['-','-','-','-','-','-','-']
    #linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1)
    plt.grid(True)
    # training loss
    marks = []
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + r'$K = $'+ str(k[i]), linewidth = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=4)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.18,  0.5])
    plt.savefig(dataset.upper() + "Convex_Mnist_train_K.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train.pdf", bbox_inches="tight")
    plt.figure(2)
    plt.grid(True)
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + r'$K = $'+ str(k[i]), linewidth = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=4)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.9,  0.947])
    #plt.ylim([0.89,  0.945])
    plt.savefig(dataset.upper() + "Convex_Mnist_test_K.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_mnist_R(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, alpha, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    glob_acc, train_acc, train_loss = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,alpha, algorithms_list, batch_size, dataset, k,personal_learning_rate)
    linestyles = ['-', '--', '-.','-', '--', '-.']
    linestyles = ['--','-','-.','--','-','-.']
    #linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1)
    plt.grid(True)
    # training loss
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + r': $R = $'+ str(loc_ep1[i]), linewidth = 0.7, color=colors[i],marker = markers[i],markevery=0.2, markersize=4)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.18,  0.5])
    plt.savefig(dataset.upper() + "Convex_Mnist_train_R.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train.pdf", bbox_inches="tight")
    plt.figure(2)
    plt.grid(True)
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + r': $R = $'+ str(loc_ep1[i]), linewidth = 0.7, color=colors[i],marker = markers[i],markevery=0.2, markersize=4)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.85,  0.95])
    #plt.ylim([0.89,  0.945])
    plt.savefig(dataset.upper() + "Convex_Mnist_test_R.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_mnist_L(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, alpha, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    glob_acc, train_acc, train_loss = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,alpha, algorithms_list, batch_size, dataset, k,personal_learning_rate)
    linestyles = ['-', '--', '-.','-', '--', '-.']
    linestyles = ['--','-','-.','--','-','-.']
    #linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","d"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1)
    plt.grid(True)
    # training loss
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + r': $\lambda = $'+ str(lamb[i]), linewidth = 0.7, color=colors[i],marker = markers[i],markevery=0.2, markersize=4)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.18,  0.5])
    plt.savefig(dataset.upper() + "Convex_Mnist_train_L.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train.pdf", bbox_inches="tight")
    plt.figure(2)
    plt.grid(True)
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + r': $\lambda = $'+ str(lamb[i]), linewidth = 0.7, color=colors[i],marker = markers[i],markevery=0.2, markersize=4)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.85,  0.95])
    #plt.ylim([0.89,  0.945])
    plt.savefig(dataset.upper() + "Convex_Mnist_test_L.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_mnist_D(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, alpha, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    glob_acc, train_acc, train_loss = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,alpha, algorithms_list, batch_size, dataset, k,personal_learning_rate)
    linestyles = ['-', '--', '-.','-', '--', '-.']
    linestyles = ['--','-','-.','--','-','-.']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1)
    plt.grid(True)
    # training loss
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + r': $|\mathcal{D}_{i}|=$'+ str(batch_size[i]), linewidth = 0.7, color=colors[i],marker = markers[i],markevery=0.2, markersize=4)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.18,  0.5])
    plt.savefig(dataset.upper() + "Convex_Mnist_train_D.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train.pdf", bbox_inches="tight")
    plt.figure(2)
    plt.grid(True)
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + r': $|\mathcal{D}_{i}|=$'+ str(batch_size[i]), linewidth = 0.7, color=colors[i],marker = markers[i],markevery=0.2, markersize=4)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy', size=14)
    plt.xlabel('Global rounds', size=14)
    plt.ylim([0.85,  0.95])
    #plt.ylim([0.89,  0.945])
    plt.savefig(dataset.upper() + "Convex_Mnist_test_D.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test.pdf", bbox_inches="tight")
    plt.close()