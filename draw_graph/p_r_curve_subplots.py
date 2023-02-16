import os
import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from options.config import Config

def p_r_subplot_3():

    classes = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Local", "Near_full", "Random", "Scratch", "Normal"]
    network = "capsnet"  # ['resnet', 'densenet', 'efficientnet', 'capsnet']
    in_dim = [16, 8]
    out_channels = [256, 512, 1024]

    data_1 = np.load('../p_r_data/p_r_data_capsnet/p_r_data_capsnet_mse_{}_{}.npz'.format(in_dim[0], out_channels[0]))
    data_2 = np.load('../p_r_data/p_r_data_capsnet/p_r_data_capsnet_mse_{}_{}.npz'.format(in_dim[0], out_channels[1]))
    data_3 = np.load('../p_r_data/p_r_data_capsnet/p_r_data_capsnet_mse_{}_{}.npz'.format(in_dim[0], out_channels[2]))

    p_1 = data_1['x']
    r_1 = data_1['y']

    p_2 = data_2['x']
    r_2 = data_2['y']

    p_3 = data_3['x']
    r_3 = data_3['y']

    for i in range(p_1.shape[2]):
        n_p_1 = []
        n_r_1 = []

        n_p_2 = []
        n_r_2 = []

        n_p_3 = []
        n_r_3 = []
        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for j in range(p_1.shape[0]):
            n_p_1.append(p_1[j][0][i])
            n_r_1.append(r_1[j][0][i])

            n_p_2.append(p_2[j][0][i])
            n_r_2.append(r_2[j][0][i])

            n_p_3.append(p_3[j][0][i])
            n_r_3.append(r_3[j][0][i])

        fig, ax = plt.subplots(1, 3, figsize=(15, 7))

        fig.suptitle("P-R curve (Network : {} Defect : {})\n".format(network, classes[i]))

        ax[0].plot(n_r_1, n_p_1, 'go', label='MSE')
        ax[0].axis([0.9, 1.0, 0.9, 1.0])
        ax[0].grid(True)
        ax[0].set_title("in_dim : {}, out_channel : {}\n".format(in_dim[0], out_channels[0]))
        ax[0].set_xlabel("recall")
        ax[0].set_ylabel("precision")
        ax[0].legend()
        for k in range(p_1.shape[0]):
            ax[0].text(n_r_1[k], n_p_1[k], 'threshold({})'.format(threshold[k]))

        ax[1].plot(n_r_2, n_p_2, 'go', label='MSE')
        ax[1].axis([0.9, 1.0, 0.9, 1.0])
        ax[1].grid(True)
        ax[1].set_title("in_dim : {}, out_channel : {}\n".format(in_dim[0], out_channels[1]))
        ax[1].set_xlabel("recall")
        ax[1].set_ylabel("precision")
        ax[1].legend()
        for p in range(p_2.shape[0]):
            ax[1].text(n_r_2[p], n_p_2[p], 'threshold({})'.format(threshold[p]))

        ax[2].plot(n_r_3, n_p_3, 'go', label='MSE')
        ax[2].axis([0.9, 1.0, 0.9, 1.0])
        ax[2].grid(True)
        ax[2].set_title("in_dim : {}, out_channel : {}\n".format(in_dim[0], out_channels[2]))
        ax[2].set_xlabel("recall")
        ax[2].set_ylabel("precision")
        ax[2].legend()
        for f in range(p_3.shape[0]):
            ax[2].text(n_r_3[f], n_p_3[f], 'threshold({})'.format(threshold[f]))

        plt.subplots_adjust(wspace=0.6)
        plt.savefig('../p_r_data/p_r_curve/p_r_data_capsnet_{}.png'.format(classes[i]))

        plt.clf()

    print('finish')

def p_r_subplot_2():
    classes = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Local", "Near_full", "Random", "Scratch", "Normal" ]
    network = "capsnet"  # ['resnet', 'densenet', 'efficientnet', 'capsnet']
    in_dim = [8, 16]
    out_channels = [256, 512, 1024]

    # loss_mse = np.load('../p_r_data/p_r_data_{}/p_r_data_{}_mse.npz'.format(network, network))
    # loss_cross = np.load('../p_r_data/p_r_data_{}/p_r_data_{}_cross.npz'.format(network, network))
    data_1 = np.load('../p_r_data/p_r_data_capsnet/p_r_data_capsnet_mse_{}_{}.npz'.format(in_dim[0], out_channels[0]))
    data_2 = np.load('../p_r_data/p_r_data_capsnet/p_r_data_capsnet_mse_{}_{}.npz'.format(in_dim[1], out_channels[0]))


    p_1 = data_1['x']
    r_1 = data_1['y']

    p_2 = data_2['x']
    r_2 = data_2['y']


    for i in range(p_1.shape[2]):
        k=0
        p=0

        n_p_1 = []
        n_r_1 = []

        n_p_2 = []
        n_r_2 = []

        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for j in range(p_1.shape[0]):

            n_p_1.append(p_1[j][0][i])
            n_r_1.append(r_1[j][0][i])

            n_p_2.append(p_2[j][0][i])
            n_r_2.append(r_2[j][0][i])

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        fig.suptitle("P-R curve (Network : {} Defect : {})".format(network, classes[i]))


        ax[0].plot(n_r_1 ,n_p_1, 'go', label='MSE')
        ax[0].axis([0.9, 1.0, 0.9, 1.0])
        ax[0].grid(True)
        ax[0].set_title("in_dim : {}, out_channel : {}".format(in_dim[0], out_channels[0]))
        ax[0].set_xlabel("recall")
        ax[0].set_ylabel("precision")
        ax[0].legend()
        for k in range(p_1.shape[0]):
            ax[0].text(n_r_1[k], n_p_1[k], 'threshold({})'.format(threshold[k]))

        ax[1].plot(n_r_2, n_p_2, 'go', label='MSE')
        ax[1].axis([0.9, 1.0, 0.9, 1.0])
        ax[1].grid(True)
        ax[1].set_title("in_dim : {}, out_channel : {}".format(in_dim[1], out_channels[0]))
        ax[1].set_xlabel("recall")
        ax[1].set_ylabel("precision")
        ax[1].legend()
        for p in range(p_2.shape[0]):
            ax[1].text(n_r_2[p], n_p_2[p], 'threshold({})'.format(threshold[p]))

        plt.subplots_adjust(wspace=0.4)
        plt.savefig('../p_r_data/p_r_curve/p_r_data_capsnet_{}.png'.format(classes[i]))

        plt.clf()

    print('finish')


if __name__ =='__main__':

    # p_r_subplot_2()

    classes = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Local", "Near_full", "Random", "Scratch", "Normal"]
    network = "capsnet"  # ['resnet', 'densenet', 'efficientnet', 'capsnet']
    in_dim = [8, 16]
    out_channels = [256, 512, 1024]

    data_1 = np.load('../p_r_data/p_r_data_capsnet/p_r_data_capsnet_mse_{}_{}.npz'.format(in_dim[0], out_channels[2]))
    data_2 = np.load('../p_r_data/p_r_data_capsnet/p_r_data_capsnet_mse_{}_{}.npz'.format(in_dim[1], out_channels[2]))

    p_1 = data_1['x']
    r_1 = data_1['y']

    p_2 = data_2['x']
    r_2 = data_2['y']


    for i in range(p_1.shape[2]):
        n_p_1 = []
        n_r_1 = []

        n_p_2 = []
        n_r_2 = []

        n_p_3 = []
        n_r_3 = []
        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        for j in range(p_1.shape[0]):
            n_p_1.append(p_1[j][0][i])
            n_r_1.append(r_1[j][0][i])

            n_p_2.append(p_2[j][0][i])
            n_r_2.append(r_2[j][0][i])

        fig, ax = plt.subplots(1, 2, figsize=(12, 7))

        fig.suptitle("P-R curve (Network : {} Defect : {})\n".format(network, classes[i]))

        ax[0].plot(n_r_1, n_p_1, 'go', label='MSE')
        ax[0].axis([0.9, 1.0, 0.9, 1.0])
        ax[0].grid(True)
        ax[0].set_title("in_dim : {}, out_channel : {}\n".format(in_dim[0], out_channels[2]))
        ax[0].set_xlabel("recall")
        ax[0].set_ylabel("precision")
        ax[0].legend()
        for k in range(p_1.shape[0]):
            ax[0].text(n_r_1[k], n_p_1[k], 'threshold({})'.format(threshold[k]))

        ax[1].plot(n_r_2, n_p_2, 'go', label='MSE')
        ax[1].axis([0.9, 1.0, 0.9, 1.0])
        ax[1].grid(True)
        ax[1].set_title("in_dim : {}, out_channel : {}\n".format(in_dim[1], out_channels[2]))
        ax[1].set_xlabel("recall")
        ax[1].set_ylabel("precision")
        ax[1].legend()
        for p in range(p_2.shape[0]):
            ax[1].text(n_r_2[p], n_p_2[p], 'threshold({})'.format(threshold[p]))


        plt.subplots_adjust(wspace=0.6)
        plt.savefig('../p_r_data/p_r_curve/p_r_data_capsnet_{}.png'.format(classes[i]))

        plt.clf()

    print('finish')