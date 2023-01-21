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

    loss_mse = np.load('../p_r_data/p_r_data_{}_mse.npz'.format(config.opt.network_name))
    loss_cross = np.load('../p_r_data/p_r_data_{}_cross.npz'.format(config.opt.network_name))

    p_mse = loss_mse['x']
    r_mse = loss_mse['y']

    p_cross = loss_cross['x']
    r_cross = loss_cross['y']

    print("p_mse.shape : {}".format(p_mse.shape))
    print("r_mse.shape : {}".format(r_mse.shape))
    print("p_cross.shape : {}".format(p_cross.shape))
    print("r_cross.shape : {}".format(r_cross.shape))


    for i in range(p_mse.shape[2]):
        p_m = []
        p_c = []

        r_m = []
        r_c = []

        for j in range(p_mse.shape[0]):
            p_m.append(p_mse[j][0][i])
            r_m.append(r_mse[j][0][i])

            p_c.append(p_cross[j][0][i])
            r_c.append(r_cross[j][0][i])

        plt.title("P-R curve (Defect : {})".format(classes[i]))
        plt.plot(r_m, p_m, color='blue', label='MSE')
        plt.plot(r_c, p_c, color='red', label='Cross Entropy')
        plt.grid(True)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend(loc='upper right')
        plt.savefig('../p_r_data/p_r_curve/p_r_curve_{}.png'.format(classes[i]))

        plt.clf()

        print('p_r_curve_{}.png'.format(classes[i]))
