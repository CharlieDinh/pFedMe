import matplotlib.pyplot as plt
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


def simple_read_data(loc_ep, alg):
    hf = h5py.File("./results/"+'{}_{}.h5'.format(alg, loc_ep), 'r')
    rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
    rs_train_acc = np.array(hf.get('rs_train_acc')[:])
    rs_train_loss = np.array(hf.get('rs_train_loss')[:])
    return rs_train_acc, rs_train_loss, rs_glob_acc

def get_training_data_value(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[],hyper_learning_rate=[],algorithms_list=[], batch_size=0, dataset=""):
    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        if(lamb[i] > 0):
            algorithms_list[i] = algorithms_list[i] + "_prox_" + str(lamb[i])
            algs_lbl[i] = algs_lbl[i] + "_prox"

        string_learning_rate = str(learning_rate[i])
        
        if(algorithms_list[i] == "fedfedl"):
            string_learning_rate = string_learning_rate + "_" +str(hyper_learning_rate[i])
        algorithms_list[i] = algorithms_list[i] + \
            "_" + string_learning_rate + "_" + str(num_users) + \
            "u" + "_" + str(batch_size[i]) + "b"
       
        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(loc_ep1[i], dataset + algorithms_list[i]))[:, :Numb_Glob_Iters]
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
def plot_data_with_inset(plt, title="", data=[], linestyles=[], labels=[], x_label="", y_label="",
                         legend_loc="lower right", plot_from=0, plot_to=-1,
                         axins_loc=None, axins_axis_visible=True, axins_zoom_factor=25,
                         axins_x_y_lims=[0, 0, -1, -1], axins_aspect=1500,
                         inset_loc1=1, inset_loc2=2, output_path=None):
    """
    If inset plot is not to be drawn, set axins_loc to None.
    If the highest and lowest thresholds in the inset plot are to be automatically determined,
    set y1, y2 to -1, -1.
    """

    plot_to = plot_to if plot_to != -1 else len(data[0])

    # Create a new figure with a default 111 subplot
    fig, ax = plt.subplots()

    # Plot the entire data
    for i in range(len(data)):
        ax.plot(range(plot_from, plot_to), data[i][plot_from:plot_to],
                linestyle=linestyles[i], label=labels[i])
    plt.legend(loc=legend_loc)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title.upper())

    # Decide to plot the inset plot or not
    if axins_loc is not None:

        # Create a zoomed portion of the original plot
        axins = zoomed_inset_axes(ax, axins_zoom_factor, loc=axins_loc)
        for i in range(len(data)):
            axins.plot(range(plot_from, plot_to), data[i][plot_from:plot_to],
                       linestyle=linestyles[i], label=labels[i])
        # specify the limits (four bounding box corners of the inset plot)
        x1, x2, y1, y2 = axins_x_y_lims

        # Automatically set the highest and lowest thresholds in the inset plot
        if (y1 == -1):
            y1 = min([min(dataset[x1:x2]) for dataset in data]) - 0.0005
            y2 = max([max(dataset[x1:x2]) for dataset in data]) + 0.0005

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_aspect(axins_aspect)
        plt.yticks(visible=axins_axis_visible)
        plt.xticks(visible=axins_axis_visible)

        # Choose which corners of the inset plot the inset markers are attachted to
        mark_inset(ax, axins, loc1=inset_loc1,
                   loc2=inset_loc2, fc="none", ec="0.6")

    # Save the figure
    if output_path is not None:
        plt.savefig(output_path)

def plot_data_with_inset_two_figures(plt, title="", data=[], linestyles=[], labels=[], x_label="", y_label="",
                         legend_loc="lower right", plot_from=0, plot_to=-1,
                         axins_loc=None, axins_axis_visible=True, axins_zoom_factor=25,
                         axins_x_y_lims=[0, 0, -1, -1], axins_aspect=1500,
                         inset_loc1=1, inset_loc2=2, output_path=None):
    """
    If inset plot is not to be drawn, set axins_loc to None.
    If the highest and lowest thresholds in the inset plot are to be automatically determined,
    set y1, y2 to -1, -1.
    """

    plot_to = plot_to if plot_to != -1 else len(data[0])

    # Create a new figure with a default 111 subplot
    # fig, ax = plt.subplots()
    ax = plt

    # Plot the entire data
    for i in range(len(data)):
        ax.plot(range(plot_from, plot_to), data[i][plot_from:plot_to],
                linestyle=linestyles[i], label=labels[i])
    plt.legend(loc=legend_loc)
    plt.set_ylabel(y_label)
    plt.set_xlabel(x_label)
    plt.set_title(title.upper())

    # Decide to plot the inset plot or not
    if axins_loc is not None:

        # Create a zoomed portion of the original plot
        axins = zoomed_inset_axes(ax, axins_zoom_factor, loc=axins_loc)
        for i in range(len(data)):
            axins.plot(range(plot_from, plot_to), data[i][plot_from:plot_to],
                       linestyle=linestyles[i], label=labels[i])
        # specify the limits (four bounding box corners of the inset plot)
        x1, x2, y1, y2 = axins_x_y_lims

        # Automatically set the highest and lowest thresholds in the inset plot
        if (y1 == -1):
            y1 = min([min(dataset[x1:x2]) for dataset in data]) - 0.0005
            y2 = max([max(dataset[x1:x2]) for dataset in data]) + 0.0005

        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_aspect(axins_aspect)
        # plt.set_yticks(visible=axins_axis_visible)
        # plt.set_xticks(visible=axins_axis_visible)

        # Choose which corners of the inset plot the inset markers are attachted to
        mark_inset(ax, axins, loc1=inset_loc1,
                   loc2=inset_loc2, fc="none", ec="0.6")

    # Save the figure
    if output_path is not None:
        plt.savefig(output_path)

def plot_summary_one_figure(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[], batch_size=0, dataset=""):
    
    
    Numb_Algs = len(algorithms_list)

    # get training data from file
    glob_acc, train_acc, train_loss = get_training_data_value(num_users,loc_ep1, Numb_Glob_Iters, lamb, learning_rate, algorithms_list, batch_size, dataset)
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    algs_lbl = algorithms_list.copy()

    ### setup value for mini-figure
    num_global_update = len(train_loss[0]) - 1
    range_plot = 100
    start_zoom_index = num_global_update - range_plot

    # get min max in range
    range_train_acc, range_train_loss, range_glob_acc = [], [], []
    for i in range(Numb_Algs):
        range_train_acc.append(train_acc[i][start_zoom_index:].min())
        range_train_acc.append(train_acc[i][start_zoom_index:].max())
        range_train_loss.append(train_loss[i][start_zoom_index:].min())
        range_train_loss.append(train_loss[i][start_zoom_index:].max())
        range_glob_acc.append(glob_acc[i][start_zoom_index:].min())
        range_glob_acc.append(glob_acc[i][start_zoom_index:].max())

    plt.figure(1)
    data, lstyles, labels = get_data_label_style(input_data=train_acc, linestyles=linestyles, algs_lbl=algs_lbl, lamb=lamb, loc_ep1=loc_ep1, batch_size=batch_size)
    plot_data_with_inset(plt, title=dataset.upper(), data=data, linestyles=lstyles, labels=labels,
                         x_label="Global rounds", y_label="Training accuracy", axins_loc=10,
                         axins_axis_visible=True, axins_zoom_factor=110, axins_x_y_lims=[start_zoom_index, num_global_update, -1, max(range_train_acc)],
                         axins_aspect=15000, inset_loc1=1, inset_loc2=2,
                         output_path=dataset.upper() + str(loc_ep1[1]) + 'train_acc.png')

    plt.figure(2)
    data, lstyles, labels = get_data_label_style(input_data=train_loss, linestyles=linestyles, algs_lbl=algs_lbl, lamb=lamb, loc_ep1=loc_ep1, batch_size=batch_size)

    # Note:
    # If plotting 800 global rounds: set axins_zoom_factor = 200,  axins_aspect = 8000
    # If plotting 1500 global rounds: set axins_zoom_factor = 375,  axins_aspect = 15000
    plot_data_with_inset(plt, title=dataset.upper(), data=data, linestyles=lstyles, labels=labels,
                         x_label="Global rounds", y_label="Training loss", axins_loc=7, legend_loc="upper right",
                         axins_axis_visible=True, axins_zoom_factor=375, axins_x_y_lims=[start_zoom_index, num_global_update, -1, max(range_train_loss)],
                         plot_from=0,
                         axins_aspect=15000, inset_loc1=3, inset_loc2=4,
                         output_path=dataset.upper() + str(loc_ep1[1]) + 'train_loss.png')

    plt.figure(3)
    data, lstyles, labels = get_data_label_style(input_data=glob_acc, linestyles=linestyles, algs_lbl=algs_lbl, lamb=lamb, loc_ep1=loc_ep1, batch_size=batch_size)

    # Note:
    # If plotting 800 global rounds: set axins_zoom_factor = 250,  axins_aspect = 8000
    # If plotting 1500 global rounds: set axins_zoom_factor = 90,  axins_aspect = 12000
    plot_data_with_inset(plt, title=dataset.upper(), data=data, linestyles=lstyles, labels=labels,
                         x_label="Global rounds", y_label="Test accuracy", axins_loc=10,
                         axins_axis_visible=True, axins_zoom_factor=90, axins_x_y_lims=[start_zoom_index, num_global_update, -1, max(range_glob_acc)],
                         axins_aspect=12000, inset_loc1=1, inset_loc2=2,
                         output_path=dataset.upper() + str(loc_ep1[1]) + 'glob_acc.png')

def plot_two_figures_with_insets(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[],
                                 batch_size=[], dataset=""):
    Numb_Algs = len(algorithms_list)

    # get training data from file
    glob_acc, train_acc, train_loss = get_training_data_value(num_users,loc_ep1, Numb_Glob_Iters, lamb, learning_rate, algorithms_list, batch_size, dataset)
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    algs_lbl = algorithms_list.copy()

    ### setup value for mini-figure
    num_global_update = len(train_loss[0]) - 1
    range_plot = 100
    start_zoom_index = num_global_update - range_plot

    # get min max in range
    range_train_acc, range_train_loss, range_glob_acc = [], [], []
    for i in range(Numb_Algs):
        range_train_acc.append(train_acc[i][start_zoom_index:].min())
        range_train_acc.append(train_acc[i][start_zoom_index:].max())
        range_train_loss.append(train_loss[i][start_zoom_index:].min())
        range_train_loss.append(train_loss[i][start_zoom_index:].max())
        range_glob_acc.append(glob_acc[i][start_zoom_index:].min())
        range_glob_acc.append(glob_acc[i][start_zoom_index:].max())

    data, lstyles, labels = get_data_label_style(input_data=train_loss, linestyles=linestyles, algs_lbl=algs_lbl, lamb=lamb, loc_ep1=loc_ep1, batch_size=batch_size)
    
    data1 = data[:len(data) // 2]
    data2 = data[len(data) //2 :]
    labels1 = labels[:len(labels) // 2]
    labels2 = labels[len(labels) // 2:]

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,4.5))
    plot_data_with_inset_two_figures(ax1, title=dataset.upper(), data=data1, linestyles=lstyles, labels=labels1,
                         x_label="Global rounds", y_label="Training loss", axins_loc=10, legend_loc="upper right",
                         axins_axis_visible=True, axins_zoom_factor=200, axins_x_y_lims=[start_zoom_index, num_global_update, -1, max(range_train_loss)],
                         plot_from=0,
                         axins_aspect=12500, inset_loc1=3, inset_loc2=4,
                         output_path=None)

    plot_data_with_inset_two_figures(ax2, title=dataset.upper(), data=data2, linestyles=lstyles, labels=labels2,
                         x_label="Global rounds", y_label="Training loss", axins_loc=10, legend_loc="upper right",
                         axins_axis_visible=True, axins_zoom_factor=60, axins_x_y_lims=[start_zoom_index, num_global_update, -1, max(range_train_loss)],
                         plot_from=0,
                         axins_aspect=4000, inset_loc1=3, inset_loc2=4,
                         output_path=None)

    # TODO: Change the output filename
    plt.savefig("comparison.png")

    pass

def plot_summary_two_figures(num_users=100, loc_ep1=[], Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[], batch_size=0, dataset=""):
    Numb_Algs = len(algorithms_list)

    glob_acc, train_acc, train_loss = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, algorithms_list, batch_size, dataset)

    plt.figure(1)
    linestyles = ['-', '--']
    algs_lbl = ["FEDL",  "FedAvg",
                "FEDL",  "FedAvg"]
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    #min = train_loss.min()
    min = train_loss.min() - 0.01
    max = 3.5  # train_loss.max() + 0.01
    num_al = 2
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$K_l = $' + str(loc_ep1[i]))
        ax1.set_ylim([min, max])
        ax1.legend(loc='upper right')

    for i in range(num_al):
        ax2.plot(train_loss[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$K_l = $' + str(loc_ep1[i+num_al]))
        ax2.set_ylim([min, max])
        ax2.legend(loc='upper right')
    ax.set_title('FENIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Training Loss', labelpad=15)
    plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf')
    plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.png')

    plt.figure(2)
    fig = plt.figure(figsize=(10, 4.5))
    #fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    max = glob_acc.max() + 0.01
    min = 0.1
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')

    for i in range(num_al):
        ax1.plot(glob_acc[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$K_l = $' + str(loc_ep1[i]))
        ax1.set_ylim([min, max])
        ax1.legend(loc='lower right')

    for i in range(num_al):
        ax2.plot(glob_acc[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$K_l = $' + str(loc_ep1[i+num_al]))
        ax2.set_ylim([min, max])
        ax2.legend(loc='lower right')
    ax.set_title('FENIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Test Accuracy', labelpad=15)
    plt.savefig(dataset + str(loc_ep1[1]) + 'glob_acc.pdf')
    plt.savefig(dataset + str(loc_ep1[1]) + 'glob_acc.png')


def plot_summary_three_figures(num_users=100, loc_ep1=[], Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[], batch_size=0, dataset=""):
    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value(
        num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, algorithms_list, batch_size, dataset)

    plt.figure(1)

    linestyles = ['-', '--']
    algs_lbl = ["FEDL",  "FedAvg",
                "FEDL",  "FedAvg",
                "FEDL",  "FedAvg"]

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    #min = train_loss.min()
    min = train_loss.min() - 0.01
    max = 3.5  # train_loss.max() + 0.01
    num_al = 2
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$K_l = $' + str(loc_ep1[i]) + ', $B = $' + stringbatch)
        ax1.set_ylim([min, max])
        ax1.legend(loc='upper right')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax2.plot(train_loss[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$K_l = $' + str(loc_ep1[i+num_al]) + ', $B = $' + stringbatch)
        ax2.set_ylim([min, max])
        ax2.legend(loc='upper right')

    for i in range(num_al):
        stringbatch = str(batch_size[i])
        ax3.plot(train_loss[i+num_al*2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al*2] + " : " + '$K_l = $' + str(loc_ep1[i+num_al*2]) + ', $B = $' + stringbatch)
        ax3.set_ylim([min, max])
        ax3.legend(loc='upper right')

    ax.set_title('FENIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Training Loss')
    plt.savefig(dataset + str(loc_ep1[1]) +
                'train_loss.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) +
                'train_loss.png', bbox_inches='tight')

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    #min = train_loss.min()
    min = glob_acc.min() + 0.2
    max = glob_acc.max() + 0.01  # train_loss.max() + 0.01
    num_al = 2
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        ax1.plot(glob_acc[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$K_l = $' + str(loc_ep1[i]) + ', $B = $' + stringbatch)
        ax1.set_ylim([min, max])
        ax1.legend(loc='lower right')

    for i in range(num_al):
        ax2.plot(glob_acc[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$K_l = $' + str(loc_ep1[i+num_al]) + ', $B = $' + stringbatch)
        ax2.set_ylim([min, max])
        ax2.legend(loc='lower right')

    for i in range(num_al):
        ax3.plot(glob_acc[i+num_al*2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al*2] + " : " + '$K_l = $' + str(loc_ep1[i+num_al*2]) + ', $B = $' + stringbatch)
        ax3.set_ylim([min, max])
        ax3.legend(loc='lower right')

    ax.set_title('FENIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Testing Accuracy', labelpad=15)
    plt.savefig(dataset + str(loc_ep1[1]) +
                'test_accu.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) +
                'test_accu.png', bbox_inches='tight')


def plot_summary_three_figures_batch(num_users=100, loc_ep1=[], Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[], batch_size=0, dataset=""):
    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value(
        num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, algorithms_list, batch_size, dataset)

    plt.figure(1)

    linestyles = ['-', '--']
    algs_lbl = ["FEDL",  "FedAvg",
                "FEDL",  "FedAvg",
                "FEDL",  "FedAvg"]

    print("global accurancy")
    for i in range(6):
        print(glob_acc[i].max())
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    #min = train_loss.min()
    min = train_loss.min() - 0.01
    max = 0.7  # train_loss.max() + 0.01
    num_al = 2
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$K_l = $' + str(loc_ep1[i]) + ', $B = $' + stringbatch)
        ax1.set_ylim([min, max])
        ax1.legend(loc='upper right')

    for i in range(num_al):
        stringbatch = str(batch_size[i+num_al])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(train_loss[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$K_l = $' + str(loc_ep1[i+num_al]) + ', $B = $' + stringbatch)
        ax2.set_ylim([min, max])
        ax2.legend(loc='upper right')

    for i in range(num_al):
        stringbatch = str(batch_size[i+num_al*2])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax3.plot(train_loss[i+num_al*2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al*2] + " : " + '$K_l = $' + str(loc_ep1[i+num_al*2]) + ', $B = $' + stringbatch)
        ax3.set_ylim([min, max])
        ax3.legend(loc='lower right')

    ax.set_title('MNIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Training Loss')
    plt.savefig(dataset + str(loc_ep1[1]) +
                'train_loss.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) +
                'train_loss.png', bbox_inches='tight')

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    min = 0.8
    max = glob_acc.max() + 0.01  # train_loss.max() + 0.01
    num_al = 2
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(glob_acc[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$K_l = $' + str(loc_ep1[i]) + ', $B = $' + stringbatch)
        ax1.set_ylim([min, max])
        ax1.legend(loc='lower right')

    for i in range(num_al):
        stringbatch = str(batch_size[i+num_al])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(glob_acc[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$K_l = $' + str(loc_ep1[i+num_al]) + ', $B = $' + stringbatch)
        ax2.set_ylim([min, max])
        ax2.legend(loc='lower right')

    for i in range(num_al):
        stringbatch = str(batch_size[i+num_al*2])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax3.plot(glob_acc[i+num_al*2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al*2] + " : " + '$K_l = $' + str(loc_ep1[i+num_al*2]) + ', $B = $' + stringbatch)
        ax3.set_ylim([min, max])
        ax3.legend(loc='upper right')

    ax.set_title('MNIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Testing Accuracy', labelpad=15)
    plt.savefig(dataset + str(loc_ep1[1]) +
                'testing_accuracy.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) +
                'testing_accuracy.png', bbox_inches='tight')


def plot_summary(num_users=100, loc_ep1=[], Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[], batch_size=0, dataset=""):

    #+'$\mu$'\
    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value(
        num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, algorithms_list, batch_size, dataset)

    plt.figure(1)
    linestyles = ['-', '--', '-.', '-', '--', '-.']
    algs_lbl = ["FedProxVR_Sarah", "FedProxVR_Svrg", "FedAvg",
                "FedProxVR_Sarah", "FedProxVR_Svrg", "FedAvg"]
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    max = 0.6
    min = train_loss.min() - 0.001
    num_al = 3
    interation_start = 200
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')

    for i in range(num_al):
        ax2.plot(train_loss[i, interation_start:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$\mu = $' + str(lamb[i]))
        ax2.set_ylim([min, max])
        ax2.legend()
        ax2.set_title("MNIST: " + r'$\beta = 7,$' + r'$\tau = 20$', y=1.02)

    for i in range(num_al):
        ax1.plot(train_loss[i+num_al, interation_start:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$\mu = $' + str(lamb[i]))
        ax1.set_ylim([min, max])
        ax1.legend()
        ax1.set_title("MNIST: " + r'$\beta = 5,$' + r'$\tau = 10$', y=1.02)

    ax.set_xlabel('Number of Global Iterations')
    ax.set_ylabel('Training Loss', labelpad=15)
    plt.savefig('train_loss.pdf')
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_loss.png')

    plt.figure(2)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    max = glob_acc.max() + 0.01
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')

    for i in range(3):
        ax2.plot(glob_acc[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$\mu = $' + str(lamb[i]))
        ax2.set_ylim([0.8, max])
        ax2.legend()
        ax2.set_title("MNIST: " + r'$\beta = 7,$' + r'$\tau = 20$', y=1.02)

    for (i) in range(3):
        ax1.plot(glob_acc[i+3, 1:], linestyle=linestyles[i+3],
                 label=algs_lbl[i] + " : " + '$\mu = $' + str(lamb[i]))
        ax1.set_title("MNIST: " + r'$\beta = 5,$' + r'$\tau = 10$', y=1.02)
        ax1.set_ylim([0.8, max])
        ax1.legend()
    ax.set_xlabel('Number of Global Iterations')
    ax.set_ylabel('Test Accuracy', labelpad=15)
    plt.savefig('glob_acc.pdf')
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'glob_acc.png')
    
def plot_summary_one_figure2(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[],hyper_learning_rate=[], algorithms_list=[], batch_size=0, dataset = ""):
    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value(
        num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,hyper_learning_rate, algorithms_list, batch_size, dataset)
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
    #plt.ylim([0.8, glob_acc.max()])
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_acc.png')
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
    plt.ylim([train_loss.min(), 0.5])
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'train_loss.png')
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf')
    plt.figure(3)
    for i in range(Numb_Algs):
        plt.plot(glob_acc[i, start:], linestyle=linestyles[i],
                 label=algorithms_list[i]+str(lamb[i])+"_"+str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b")
        #plt.plot(glob_acc1[i, 1:], label=algs_lbl1[i])
    plt.legend(loc='lower right')
    #plt.ylim([0.6, glob_acc.max()])
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    plt.savefig(dataset.upper() + str(loc_ep1[1]) + 'glob_acc.png')
    #plt.savefig(dataset + str(loc_ep1[1]) + 'glob_acc.pdf')

def get_max_value_index(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[], batch_size=0, dataset=""):
    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value(
        num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, algorithms_list, batch_size, dataset)
    for i in range(Numb_Algs):
        print("Algorithm: ", algorithms_list[i], "Max testing Accurancy: ", glob_acc[i].max(
        ), "Index: ", np.argmax(glob_acc[i]), "local update:", loc_ep1[i])

def plot_summary_synthetic(num_users=100, loc_ep1=[], Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[]):       
    #+'$\mu$'
    Numb_Algs = len(algorithms_list)
    train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
    glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
    algs_lbl = algorithms_list.copy()
    for i in range(Numb_Algs):
        if(lamb[i] > 0):
            algorithms_list[i] = algorithms_list[i] + "_prox_" + str(lamb[i])
            algs_lbl[i] = algs_lbl[i] + "_prox"
        algorithms_list[i] = algorithms_list[i] + "_" + str(learning_rate[i]) + "_" + str(num_users) + "u"
        train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
            simple_read_data(loc_ep1[i], DATA_SET + algorithms_list[i]))[:, :Numb_Glob_Iters]
        algs_lbl[i] = algs_lbl[i]

    plt.figure(1)
    linestyles = ['-', '--', '-.', '-', '--', '-.']
    #color = ['b', 'r', 'g', 'y']
    #plt.subplot(121)
    #for i in range(Numb_Algs):
    #    plt.plot(train_acc[i, 1:], linestyle=linestyles[i],
    #plt.plot(train_acc1[i, 1:], label=algs_lbl1[i])
    #lt.legend(loc='best')
    #plt.ylabel('Training Accuracy')
    #plt.xlabel('Number of Global Iterations')
    #plt.title(DATA_SET)
    #plt.savefig('train_acc.png')
    #fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    min = train_loss.min() - 0.01
    #min = 0.14
    algs_lbl = ["FedProxVR_Svrg", "FedProxVR_Svrg", "FedProxVR_Svrg", "FedProxVR_Svrg",
                "FedProxVR_Sarah", "FedProxVR_Sarah", "FedProxVR_Sarah", "FedProxVR_Sarah"]
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    num = 4
    for i in range(num):
        ax2.plot(train_loss[i, 1:], linestyle=linestyles[i], label=algs_lbl[i] + " : " + '$\mu = $' + str(lamb[i]))
        ax2.set_ylim([min, 1.2])
        ax2.legend()
        #ax2.set_title("Synthetic_0_0 " + r'$\beta = 15,$' + r'$\tau = 20$', y=1.02)
    
    for (i) in range(num):
        ax1.plot(train_loss[i+num, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i+num] + " : " + '$\mu = $' + str(lamb[i]))
        ax1.set_ylim([min, 1.2])
        ax1.legend()
        #ax1.set_title("Synthetic_0_0 : " + r'$\beta = 15,$' + r'$\tau = 20$', y=1.02)
    ax.set_title("Synthetic: 100 users, " + r'$\beta = 7,$' +
                  r'$\tau = 20$', y=1.02)
    ax.set_xlabel('Number of Global Iterations')
    ax.set_ylabel('Training Loss', labelpad=15)
    plt.savefig('train_loss.pdf')

    plt.figure(2)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    max = glob_acc.max() + 0.01
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num):
        ax2.plot(glob_acc[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$\mu = $' + str(lamb[i]))
        #ax2.set_ylim([0.8, max])
        ax1.set_ylim([0.5, max])
        ax2.legend()
        ax2.set_title("FASHION MNIST: " + r'$\beta = 7,$' +
                      r'$\tau = 50$', y=1.02)
    for (i) in range(num):
        ax1.plot(glob_acc[i+num, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i+num] + " : " + '$\mu = $' + str(lamb[i]))
        ax1.set_title("FASHION MNIST: " + r'$\beta = 15,$' + r'$\tau = 20$', y=1.02)
        ax1.set_ylim([0.5, max])
        ax1.legend()
    ax.set_xlabel('Number of Global Iterations')
    ax.set_ylabel('Test Accuracy', labelpad=15)
    plt.savefig('glob_acc.png')



def plot_summary_mnist(num_users=100, loc_ep1=[], Numb_Glob_Iters=10, lamb=[], learning_rate=[],hyper_learning_rate=[], algorithms_list=[], batch_size=0, dataset=""):
    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value(
        num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate,hyper_learning_rate, algorithms_list, batch_size, dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "loss:", glob_acc[i].max())
    plt.figure(1)

    linestyles = ['-', '--', '-.', ':']
    algs_lbl = ["FEDL",  "FedAvg",
                "FEDL",  "FedAvg",
                "FEDL",  "FedAvg",
                "FEDL",  "FEDL"]

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    #min = train_loss.min()
    min = train_loss.min() - 0.01
    max = 0.5  # train_loss.max() + 0.01
    num_al = 2
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : "  + '$B = $' + stringbatch+ ', $\eta = $'+ str(hyper_learning_rate[i]))
        ax1.set_ylim([min, max])
        ax1.legend(loc='upper right')

    for i in range(num_al):
        stringbatch = str(batch_size[i+2])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(train_loss[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : "  + '$B = $' + stringbatch+ ', $\eta = $'+ str(hyper_learning_rate[i+num_al]))
        ax2.set_ylim([min, max])
        ax2.legend(loc='upper right')

    for i in range(4):
        stringbatch = str(batch_size[i+4])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax3.plot(train_loss[i+num_al*2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al*2] + " : "  + '$B = $' + stringbatch+ ', $\eta = $'+ str(hyper_learning_rate[i+num_al*2]))
        ax3.set_ylim([min, max])
        ax3.legend(loc='upper right')

    ax.set_title('MNIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Training Loss',x=1.03)
    plt.savefig(dataset + str(loc_ep1[1]) +
                'train_loss.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) +
                'train_loss.png', bbox_inches='tight')

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    #min = train_loss.min()
    min = 0.8
    max = glob_acc.max() + 0.01  # train_loss.max() + 0.01
    num_al = 2
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(glob_acc[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : "  + '$B = $' + stringbatch + ', $\eta = $'+ str(hyper_learning_rate[i]))
        ax1.set_ylim([min, max])
        ax1.legend(loc='lower right')

    for i in range(num_al):
        stringbatch = str(batch_size[i+2])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(glob_acc[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : "  + '$B = $' + stringbatch+ ', $\eta = $'+ str(hyper_learning_rate[i+num_al*1]))
        ax2.set_ylim([min, max])
        ax2.legend(loc='lower right')

    for i in range(4):
        stringbatch = str(batch_size[i+4])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax3.plot(glob_acc[i+num_al*2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al*2] + " : " + '$B = $' + stringbatch + ', $\eta = $'+ str(hyper_learning_rate[i+num_al*2]))
        ax3.set_ylim([min, max])
        ax3.legend(loc='lower right')

    ax.set_title('MNIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Testing Accuracy', labelpad=15)
    plt.savefig(dataset + str(loc_ep1[1]) +
                'test_accu.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) +
                'test_accu.png', bbox_inches='tight')


def plot_summary_nist(num_users=100, loc_ep1=[], Numb_Glob_Iters=10, lamb=[], learning_rate=[], hyper_learning_rate=[], algorithms_list=[], batch_size=0, dataset=""):
    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value(
        num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, hyper_learning_rate, algorithms_list, batch_size, dataset)
    for i in range(Numb_Algs):
        print(algorithms_list[i], "loss:", glob_acc[i].max())
    plt.figure(1)

    linestyles = ['-', '--', '-.', ':']
    algs_lbl = ["FEDL","FedAvg", "FEDL",
                "FEDL", "FedAvg", "FEDL",
                "FEDL", "FedAvg", "FEDL"]
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    #min = train_loss.min()
    min = train_loss.min() - 0.01
    max = 2.5  # train_loss.max() + 0.01
    num_al = 3
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(train_loss[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(hyper_learning_rate[i]) + ', $K_l = $' + str(loc_ep1[i]))
        ax1.set_ylim([min, max])
        ax1.legend(loc='upper right')

    for i in range(num_al):
        stringbatch = str(batch_size[i+num_al])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(train_loss[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(hyper_learning_rate[i+num_al]) + ', $K_l = $' + str(loc_ep1[i+ num_al]))
        ax2.set_ylim([min, max])
        ax2.legend(loc='upper right')

    for i in range(num_al):
        stringbatch = str(batch_size[i+num_al*2])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax3.plot(train_loss[i+num_al*2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al*2] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(hyper_learning_rate[i+num_al*2]) + ', $K_l = $' + str(loc_ep1[i + num_al*2]))
        ax3.set_ylim([min, max])
        ax3.legend(loc='upper right')

    ax.set_title('FENIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Training Loss')
    plt.savefig(dataset + str(loc_ep1[1]) +
                'train_loss.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) +
                'train_loss.png', bbox_inches='tight')

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    #min = train_loss.min()
    num_al = 3
    min = 0.5
    max = glob_acc.max() + 0.01  # train_loss.max() + 0.01
# Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off',
                   bottom='off', left='off', right='off')
    for i in range(num_al):
        stringbatch = str(batch_size[i])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax1.plot(glob_acc[i, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(hyper_learning_rate[i]) + ', $K_l = $' + str(loc_ep1[i]))
        ax1.set_ylim([min, max])
        ax1.legend(loc='lower right')

    for i in range(num_al):
        stringbatch = str(batch_size[i+num_al])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax2.plot(glob_acc[i+num_al, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(hyper_learning_rate[i+num_al*1]) + ', $K_l = $' + str(loc_ep1[i + num_al]))
        ax2.set_ylim([min, max])
        ax2.legend(loc='lower right')

    for i in range(num_al):
        stringbatch = str(batch_size[i+num_al*2])
        if(stringbatch == '0'):
            stringbatch = '$\infty$'
        ax3.plot(glob_acc[i+num_al*2, 1:], linestyle=linestyles[i],
                 label=algs_lbl[i + num_al*2] + " : " + '$B = $' + stringbatch + ', $\eta = $' + str(hyper_learning_rate[i+num_al*2]) + ', $K_l = $' + str(loc_ep1[i+ 2*num_al]))
        ax3.set_ylim([min, max])
        ax3.legend(loc='lower right')

    ax.set_title('FENIST', y=1.02)
    ax.set_xlabel('Global rounds ' + '$K_g$')
    ax.set_ylabel('Testing Accuracy', labelpad=15)
    plt.savefig(dataset + str(loc_ep1[1]) +
                'test_accu.pdf', bbox_inches='tight')
    plt.savefig(dataset + str(loc_ep1[1]) +
                'test_accu.png', bbox_inches='tight')
