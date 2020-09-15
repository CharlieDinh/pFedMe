import matplotlib.pyplot as plt
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import StrMethodFormatter
import os
plt.rcParams.update({'font.size': 14})

def simple_read_data(alg):
    print(alg)
    hf = h5py.File("./results/"+'{}.h5'.format(alg), 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    return rs_train_acc, rs_train_loss, rs_glob_acc

def get_training_data_value(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[],beta=[],algorithms_list=[], batch_size=[], dataset="", k= [] , personal_learning_rate = []):
    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        string_learning_rate = str(learning_rate[i])  
        string_learning_rate = string_learning_rate + "_" +str(beta[i]) + "_" +str(lamb[i])
        if(algorithms_list[i] == "pFedMe" or algorithms_list[i] == "pFedMe_p"):
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b" + "_" +str(loc_ep1[i]) + "_"+ str(k[i])  + "_"+ str(personal_learning_rate[i])
        else:
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b"  "_" +str(loc_ep1[i])

        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(dataset +"_"+ algorithms_list[i] + "_avg"))[:, :Numb_Glob_Iters]
        algs_lbl[i] = algs_lbl[i]
    return glob_acc, train_acc, train_loss

def get_all_training_data_value(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=0, learning_rate=0,beta=0,algorithms="", batch_size=0, dataset="", k= 0 , personal_learning_rate =0 ,times = 5):
    train_acc = np.zeros((times, Numb_Glob_Iters))
    train_loss = np.zeros((times, Numb_Glob_Iters))
    glob_acc = np.zeros((times, Numb_Glob_Iters))
    algorithms_list  = [algorithms] * times
    for i in range(times):
        string_learning_rate = str(learning_rate)  
        string_learning_rate = string_learning_rate + "_" +str(beta) + "_" +str(lamb)
        if(algorithms == "pFedMe" or algorithms == "pFedMe_p"):
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size) + "b" + "_" +str(loc_ep1) + "_"+ str(k)  + "_"+ str(personal_learning_rate) +  "_" +str(i) 
        else:
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size) + "b"  "_" +str(loc_ep1) +  "_" +str(i)

        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(dataset +"_"+ algorithms_list[i]))[:, :Numb_Glob_Iters]
    return glob_acc, train_acc, train_loss


def get_data_label_style(input_data = [], linestyles= [], algs_lbl = [], lamb = [], loc_ep1 = 0, batch_size =0):
    data, lstyles, labels = [], [], []
    for i in range(len(algs_lbl)):
        data.append(input_data[i, ::])
        lstyles.append(linestyles[i])
        labels.append(algs_lbl[i]+str(lamb[i])+"_" +
                      str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b")

    return data, lstyles, labels

def average_data(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb="", learning_rate="", beta="", algorithms="", batch_size=0, dataset = "", k = "", personal_learning_rate = "", times = 5):
    if(algorithms == "PerAvg"):
        algorithms = "PerAvg_p"
    glob_acc, train_acc, train_loss = get_all_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms, batch_size, dataset, k, personal_learning_rate,times)
    glob_acc_data = np.average(glob_acc, axis=0)
    train_acc_data = np.average(train_acc, axis=0)
    train_loss_data = np.average(train_loss, axis=0)
    # store average value to h5 file
    max_accurancy = []
    for i in range(times):
        max_accurancy.append(glob_acc[i].max())
    
    print("std:", np.std(max_accurancy))
    print("Mean:", np.mean(max_accurancy))

    alg = dataset + "_" + algorithms
    alg = alg + "_" + str(learning_rate) + "_" + str(beta) + "_" + str(lamb) + "_" + str(num_users) + "u" + "_" + str(batch_size) + "b" + "_" + str(loc_ep1)
    if(algorithms == "pFedMe" or algorithms == "pFedMe_p"):
        alg = alg + "_" + str(k) + "_" + str(personal_learning_rate)
    alg = alg + "_" + "avg"
    if (len(glob_acc) != 0 &  len(train_acc) & len(train_loss)) :
        with h5py.File("./results/"+'{}.h5'.format(alg,loc_ep1), 'w') as hf:
            hf.create_dataset('rs_glob_acc', data=glob_acc_data)
            hf.create_dataset('rs_train_acc', data=train_acc_data)
            hf.create_dataset('rs_train_loss', data=train_loss_data)
            hf.close()

def plot_summary_one_figure(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], beta=[], algorithms_list=[], batch_size=0, dataset = "", k = [], personal_learning_rate = []):
    Numb_Algs = len(algorithms_list)
    dataset = dataset
    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    
    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')

    #glob_acc, train_acc, train_loss = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    print("max value of test accurancy",glob_acc.max())
    plt.figure(1,figsize=(5, 5))
    MIN = train_loss.min() - 0.001
    start = 0
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for i in range(Numb_Algs):
        plt.plot(train_acc[i, 1:], linestyle=linestyles[i], label=algorithms_list[i] )
    plt.legend(loc='lower right')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Global rounds ' + '$K_g$')
    plt.title(dataset.upper())
    #plt.ylim([0.8, glob_acc.max()])
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_acc.png', bbox_inches="tight")
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')
    plt.figure(2)

    plt.grid(True)
    for i in range(Numb_Algs):
        plt.plot(train_loss[i, start:], linestyle=linestyles[i], label=algorithms_list[i] )
        #plt.plot(train_loss1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.title(dataset.upper())
    #plt.ylim([train_loss.min(), 0.5])
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_loss.png', bbox_inches="tight")
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf')
    plt.figure(3)
    plt.grid(True)
    for i in range(Numb_Algs):
        plt.plot(glob_acc[i, start:], linestyle=linestyles[i],
                 label=algorithms_list[i])
        #plt.plot(glob_acc1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='lower right')
    #plt.ylim([0.6, glob_acc.max()])
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'glob_acc.png', bbox_inches="tight")
    #plt.savefig(dataset + str(loc_ep1[1]) + 'glob_acc.pdf')

def get_max_value_index(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[], batch_size=0, dataset=""):
    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value(
        num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, algorithms_list, batch_size, dataset)
    for i in range(Numb_Algs):
        print("Algorithm: ", algorithms_list[i], "Max testing Accurancy: ", glob_acc[i].max(
        ), "Index: ", np.argmax(glob_acc[i]), "local update:", loc_ep1[i])

def get_label_name(name):
    if name.startswith("pFedMe"):
        if name.startswith("pFedMe_p"):
            return "pFedMe"+ " (PM)"
        else:
            return "pFedMe"+ " (GM)"
    if name.startswith("PerAvg"):
        return "Per-FedAvg"
    if name.startswith("FedAvg"):
        return "FedAvg"
    if name.startswith("APFL"):
        return "APFL"

def average_smooth(data, window_len=20, window='hanning'):
    results = []
    if window_len<3:
        return data
    for i in range(len(data)):
        x = data[i]
        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('numpy.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        results.append(y[window_len-1:])
    return np.array(results)

def plot_summary_one_figure_synthetic_R(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    
    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    
    
    linestyles = ['-','-','-','-.','-.','-.']
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'c', 'darkorange', 'tab:brown', 'w']
    markers = ["o","v","s","*","x","P"]
    plt.figure(1,figsize=(5, 5))
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], label=label + ": "
                 r'$R = $' +str(loc_ep1[i]), linewidth = 1, color=colors[i], marker = markers[i],markevery=0.05, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.ylim([train_loss.min() - 0.01,  2])
    #plt.ylim([0.5,  1.8])
    plt.savefig(dataset.upper() + "Non_Convex_Syn_fixR.pdf", bbox_inches="tight")

    plt.figure(2,figsize=(5, 5))
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], label=label + ": "
                 r'$R = $' +str(loc_ep1[i]), linewidth = 1, color=colors[i], marker = markers[i],markevery=0.05, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    plt.ylim([0.6,  0.86])
    #plt.ylim([0.89,  0.945])
    plt.savefig(dataset.upper() + "Non_Convex_Syn_fixR_test.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Convex_Syn_fixR.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_synthetic_K(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    
    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    
    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    
    linestyles = ['-','-','-','-.','-.','-.']
    print(lamb)
    colors = ['tab:blue', 'tab:green','darkorange', 'r', 'c', 'tab:brown', 'w']
    markers = ["o","v","s","*","x","P"]
    plt.figure(1,figsize=(5, 5))
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + ": "
                 r'$K = $' +str(k[i]), linewidth = 1, color=colors[i], marker = markers[i],markevery=0.05, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    #plt.ylim([0.5,  1.8])
    plt.ylim([train_loss.min() - 0.01,  2])
    #plt.savefig(dataset.upper() + "Non_Convex_Syn_fixK.pdf", bbox_inches="tight")
    plt.savefig(dataset.upper() + "Convex_Syn_fixK.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + ": "
                 r'$K = $' +str(k[i]), linewidth = 1, color=colors[i], marker = markers[i],markevery=0.05, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    plt.ylim([0.6,  0.86])
    plt.savefig(dataset.upper() + "Convex_Syn_fixK_test.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_synthetic_L(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    
    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    
    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    
    linestyles = ['-','-','-','-.','-.','-.']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'c', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + ": "
                 + r'$\lambda = $'+ str(lamb[i]), linewidth = 1, color=colors[i],marker = markers[i],markevery=0.05, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.ylim([0.5,  1.8])
    #plt.ylim([train_loss.min() - 0.01,  2])
    #plt.savefig(dataset.upper() + "Non_Convex_Syn_fixL.pdf", bbox_inches="tight")
    plt.savefig(dataset.upper() + "Convex_Syn_fixL.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + ": "
                 + r'$\lambda = $'+ str(lamb[i]), linewidth = 1, color=colors[i],marker = markers[i],markevery=0.05, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    plt.ylim([0.6,  0.86])
    plt.savefig(dataset.upper() + "Convex_Syn_fixL_test.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_synthetic_D(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    
    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    
    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    
    linestyles = ['-','-','-','-.','-.','-.']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'c', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + ": "
                 + r'$\lambda = $'+ str(lamb[i]), linewidth = 1, color=colors[i],marker = markers[i],markevery=0.05, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.ylim([0.5,  1.8])
    #plt.ylim([train_loss.min() - 0.01,  2])
    #plt.savefig(dataset.upper() + "Non_Convex_Syn_fixL.pdf", bbox_inches="tight")
    plt.savefig(dataset.upper() + "Convex_Syn_fixL.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + ": "
                 + r'$\lambda = $'+ str(lamb[i]), linewidth = 1, color=colors[i],marker = markers[i],markevery=0.05, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    plt.ylim([0.6,  0.86])
    plt.savefig(dataset.upper() + "Convex_Syn_fixL_test.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_synthetic_Compare(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    for i in range(Numb_Algs):
        print("max accurancy:", train_acc_[i].max())
    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    linestyles = ['-', '--', '-.','-', '--', '-.']
    linestyles = ['-','-','-','-','-','-','-']
    #linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    plt.grid(True)
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label, linewidth = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.ylim([0.5,  1.8]) # convex
    #plt.ylim([0.4,  1.8]) # non convex
    #plt.ylim([train_loss.min() - 0.01,  2])
    #plt.savefig(dataset.upper() + "Non_Convex_Syn_train_Com.pdf", bbox_inches="tight")
    plt.savefig(dataset.upper() + "Convex_Syn_train_Com.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    plt.grid(True)
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i],label=label, linewidth = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    plt.ylim([0.5,  0.86]) # convex 
    #plt.savefig(dataset.upper() + "Non_Convex_Syn_test_Com.pdf", bbox_inches="tight")
    plt.savefig(dataset.upper() + "Convex_Syn_test_Com.pdf", bbox_inches="tight")
    plt.close()


def plot_summary_one_figure_mnist_Compare(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    
    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    for i in range(Numb_Algs):
        print("max accurancy:", glob_acc_[i].max())
    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    
    linestyles = ['-', '--', '-.','-', '--', '-.']
    linestyles = ['-','-','-','-','-','-','-']
    #linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    plt.grid(True)
    # training loss
    marks = []
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label, linewidth = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    #plt.ylim([0.05,  0.6]) # non convex-case
    plt.ylim([0.19,  0.4]) # convex-case
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    plt.savefig(dataset.upper() + "Convex_Mnist_train_Com.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train_Com.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    plt.grid(True)
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label, linewidth = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    #plt.ylim([0.84,  0.98]) # non convex-case
    plt.ylim([0.88,  0.95]) # Convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_test_Com.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test_Com.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_mnist_K(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    
    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    
    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    
    linestyles = ['-','-','-','-.','-.','-.']
    #linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    plt.grid(True)
    # training loss
    marks = []
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + r': $K = $'+ str(k[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    #plt.ylim([0.05,  0.6]) # non convex-case
    plt.ylim([0.19,  0.5]) # convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_train_K.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train_K.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + r': $K = $'+ str(k[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    # plt.ylim([0.84,  0.98]) # non convex-case
    plt.ylim([0.86,  0.95]) # Convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_test_K.pdf", bbox_inches="tight")
   #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test_K.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_mnist_R(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    
    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    
    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    linestyles = ['-','-','-','-.','-.','-.']
    #linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # training loss
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + r': $R = $'+ str(loc_ep1[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    #plt.ylim([0.05,  0.6]) # non convex-case
    plt.ylim([0.17,  0.5]) # convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_train_R.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train_R.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + r': $R = $'+ str(loc_ep1[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    # plt.ylim([0.84,  0.985]) # non convex-case
    plt.ylim([0.86,  0.955]) # Convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_test_R.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test_R.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_mnist_L(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset

    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    
    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    linestyles = ['-','-','-','-.','-.','-.']
    #linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","d"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # training loss
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + r': $\lambda = $'+ str(lamb[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    #plt.ylim([0.05,  0.6]) # non convex-case
    plt.ylim([0.19,  0.5]) # convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_train_L.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train_L.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + r': $\lambda = $'+ str(lamb[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    #plt.ylim([0.84,  0.98]) # non convex-case
    plt.ylim([0.86,  0.95]) # Convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_test_L.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test_L.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_mnist_D(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    
    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    
    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    linestyles = ['-','-','-','-.','-.','-.']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # training loss
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + r': $|\mathcal{D}|=$'+ str(batch_size[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    #plt.ylim([0.05,  0.6]) # non convex-case
    plt.ylim([0.19,  0.5]) # convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_train_D.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train_D.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + r': $|\mathcal{D}|=$'+ str(batch_size[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    # plt.ylim([0.84,  0.98]) # non convex-case
    plt.ylim([0.86,  0.95]) # Convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_test_D.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test_D.pdf", bbox_inches="tight")
    plt.close()


def plot_summary_one_figure_mnist_Beta(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)   
    dataset = dataset
    
    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    
    glob_acc =  average_smooth(glob_acc_,window_len=10, window='flat')
    train_loss = average_smooth(train_loss_,window_len=10, window='flat')
    train_acc = average_smooth(train_acc_,window_len=10, window='flat')
    
    linestyles = ['-','-','-','-.','-.','-.']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # training loss
    marks = []
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + r': $\beta = $'+ str(beta[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    #plt.ylim([0.05,  0.6]) # non convex-case
    plt.ylim([0.18,  0.5]) # convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_train_Beta.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train_Beta.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + r': $\beta = $'+ str(beta[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    # plt.ylim([0.84,  0.985]) # non convex-case
    plt.ylim([0.88,  0.946]) # Convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_test_Beta.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test_Beta.pdf", bbox_inches="tight")
    plt.close()