import os
import sys
sys.path.append("..")
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import wandb

if __name__ =='__main__':

    classes = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Local", "Near_full", "Random", "Scratch", "Normal"]
    in_dim = [8, 16]
    out_channels = [256, 512, 1024]

    data_1 = np.load('../p_r_data/p_r_data_resnet/p_r_data_resnet_cross_16_1024_0.5.npz')
    # data_1 = np.load('../p_r_data/p_r_data_resnet_2/p_r_data_resnet_2_cross_16_1024_0.5.npz')
    data_2 = np.load('../p_r_data/p_r_data_densenet/p_r_data_densenet_cross_16_1024_0.5.npz')
    data_3 = np.load('../p_r_data/p_r_data_efficientnet/p_r_data_efficientnet_cross_16_1024_0.5.npz')
    data_4 = np.load('../p_r_data/p_r_data_capsnet/p_r_data_capsnet_mse_{}_{}_0.5.npz'.format(in_dim[0], out_channels[0]))


    p_r = data_1['x'] # resnet_34_precision
    r_r = data_1['y'] # resnet_34_recall

    p_d = data_2['x'] # densenet_precision
    r_d = data_2['y'] # densenet_recall

    p_e = data_3['x'] # efficientnet_precision
    r_e = data_3['y'] # efficientnet_recall

    p_c = data_4['x'] # capsulnet_precision
    r_c = data_4['y'] # capsulnet_recall

    for i in range(p_c.shape[2]):
        tp_r = []
        tr_r = []

        tp_d = []
        tr_d = []

        tp_e = []
        tr_e = []

        tp_c = []
        tr_c = []

        for j in range(p_c.shape[0]):
            tp_r = (p_r[j][0][i])
            tr_r = (r_r[j][0][i])

            tp_d = (p_d[j][0][i])
            tr_d = (r_d[j][0][i])

            tp_e = (p_e[j][0][i])
            tr_e = (r_e[j][0][i])

            tp_c = (p_c[j][0][i])
            tr_c = (r_c[j][0][i])

        fig = plt.figure()
        plt.title("P-R curve (Defect : {})".format(classes[i]))
        plt.axis([0.93, 1.01, 0.6, 1.01])

        # plt.plot(tr_r, tp_r, label='Resnet_34')
        plt.plot(tr_r, tp_r, label='Resnet_50')
        plt.plot(tr_d, tp_d, label='Densenet')
        plt.plot(tr_e, tp_e, label='Efficientnet')
        plt.plot(tr_c, tp_c, label='Capsnet')


        plt.grid(True)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend()

        plt.savefig('../p_r_data/p_r_curve/p-r_curve_{}_resnet_50.png'.format(classes[i]))
        plt.clf()