import os
import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from options.config import Config

if __name__ =='__main__':

    config = Config()

    classes = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Local", "Near_full", "Random", "Scratch", "Normal" ]
    network = "capsnet"  # ['resnet', 'densenet', 'efficientnet', 'capsnet']
    in_dim = 8 # [8, 16]
    out_channels = 256 # [256, 512, 1024]

    # loss_mse = np.load('../p_r_data/p_r_data_{}/p_r_data_{}_mse.npz'.format(network, network))
    # loss_cross = np.load('../p_r_data/p_r_data_{}/p_r_data_{}_cross.npz'.format(network, network))
    loss_cross = np.load('../p_r_data/p_r_data_capsnet/p_r_data_capsnet_mse_{}_{}.npz'.format(in_dim, out_channels))

    # p_mse = loss_mse['x']
    # r_mse = loss_mse['y']

    p_cross = loss_cross['x']
    r_cross = loss_cross['y']

    # print("p_mse.shape : {}".format(p_mse.shape))
    # print("r_mse.shape : {}".format(r_mse.shape))
    print("p_cross.shape : {}".format(p_cross.shape))
    print("r_cross.shape : {}".format(r_cross.shape))


    for i in range(p_cross.shape[2]):
        # p_m = []
        # r_m = []

        p_c = []
        r_c = []
        threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for j in range(p_cross.shape[0]):
            # p_m.append(p_mse[j][0][i])
            # r_m.append(r_mse[j][0][i])

            p_c.append(p_cross[j][0][i])
            r_c.append(r_cross[j][0][i])

        plt.figure(figsize=(10, 7))
        plt.axis([0.9, 1.0, 0.8, 1.0])
        plt.title("P-R curve (Network : {} Defect : {})".format(network, classes[i]))

        # plt.plot(r_m, p_m, 'bx', label='MSE')
        if not network == 'capsnet':
            plt.plot(r_c, p_c, 'go', label='Cross Entropy')
        elif network == 'capsnet':
            plt.plot(r_c, p_c, 'go-', label='MSE')

        for k in range(p_cross.shape[0]):
            # plt.text(r_m[k], p_m[k], 'threshold({})'.format(threshold[k]))
            plt.text(r_c[k], p_c[k], 'threshold({})'.format(threshold[k]))

        plt.grid(True)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend()

        plt.savefig('../p_r_data/p_r_curve/p_r_data_capsnet_{}_{}_{}.png'.format(in_dim, out_channels, classes[i]))

        plt.clf()

        print('p_r_data_capsnet_{}_{}_{}.png'.format(in_dim, out_channels, classes[i]))
