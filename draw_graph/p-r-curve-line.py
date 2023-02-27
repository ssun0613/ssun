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

    data_1 = np.load('../p_r_data/p_r_data_resnet/p_r_data_resnet_cross_16_1024_1.npz')
    data_2 = np.load('../p_r_data/p_r_data_densenet/p_r_data_densenet_cross_16_1024_1.npz')
    data_3 = np.load('../p_r_data/p_r_data_efficientnet/p_r_data_efficientnet_cross_16_1024_1.npz')
    data_4 = np.load('../p_r_data/p_r_data_capsnet/p_r_data_capsnet_mse_{}_{}_1.npz'.format(in_dim[1], out_channels[2]))

    p_r = data_1['x']
    r_r = data_1['y']

    p_d = data_2['x']
    r_d = data_2['y']

    p_e = data_3['x']
    r_e = data_3['y']

    p_c = data_4['x']
    r_c = data_4['y']

    for i in range(p_c.shape[2]):
        n_p_r = []
        n_r_r = []

        n_p_d = []
        n_r_d = []

        n_p_e = []
        n_r_e = []

        n_p_c = []
        n_r_c = []

        for j in range(p_c.shape[0]):
            tp_r = p_r[j][0][i]
            tr_r = r_r[j][0][i]
            flag_save_r = True

            tp_d = p_d[j][0][i]
            tr_d = r_d[j][0][i]
            flag_save_d = True

            tp_e = p_e[j][0][i]
            tr_e = r_e[j][0][i]
            flag_save_e = True

            tp_c = p_c[j][0][i]
            tr_c = r_c[j][0][i]
            flag_save_c = True

            for k in range(p_c.shape[0]):
                if tp_r < p_r[k][0][i]:
                    if tr_r < r_r[k][0][i]:
                        flag_save_r = False
                        continue

                if tp_d < p_d[k][0][i]:
                    if tr_d < r_d[k][0][i]:
                        flag_save_d = False
                        continue

                if tp_e < p_e[k][0][i]:
                    if tr_e < r_e[k][0][i]:
                        flag_save_e = False
                        continue

                if tp_c < p_c[k][0][i]:
                    if tr_c < r_c[k][0][i]:
                        flag_save_c = False
                        continue

            if flag_save_r:
                n_p_r.append(tp_r)
                n_r_r.append(tr_r)

            if flag_save_d:
                n_p_d.append(tp_d)
                n_r_d.append(tr_d)

            if flag_save_e:
                n_p_e.append(tp_e)
                n_r_e.append(tr_e)

            if flag_save_c:
                n_p_c.append(tp_c)
                n_r_c.append(tr_c)

        listsort_r = []
        for idx in range(len(n_r_r)):
            listsort_r.append([n_r_r[idx], n_p_r[idx]])
        listsort_r.sort(key=lambda x: (x[0], -x[1]))

        listsort_d = []
        for idx in range(len(n_r_d)):
            listsort_d.append([n_r_d[idx], n_p_d[idx]])
        listsort_d.sort(key=lambda x: (x[0], -x[1]))

        listsort_e = []
        for idx in range(len(n_r_e)):
            listsort_e.append([n_r_e[idx], n_p_e[idx]])
        listsort_e.sort(key=lambda x: (x[0], -x[1]))

        listsort_c = []
        for idx in range(len(n_r_c)):
            listsort_c.append([n_r_c[idx], n_p_c[idx]])
        listsort_c.sort(key=lambda x: (x[0], -x[1]))

        # listsort_r.sort(key=lambda x: (x[0], x[1]))
        # listsort_r.sort(key=lambda x: (x[0], -x[1]))
        # listsort_d.sort(key=lambda x: (x[0], -x[1]))
        # listsort_e.sort(key=lambda x: (x[0], -x[1]))
        # listsort_c.sort(key=lambda x: (x[0], -x[1]))

        r_n_p = []
        r_n_r = []
        for value in listsort_r:
            r_n_p.append(value[1])
            r_n_r.append(value[0])

        d_n_p = []
        d_n_r = []
        for value in listsort_d:
            d_n_p.append(value[1])
            d_n_r.append(value[0])

        e_n_p = []
        e_n_r = []
        for value in listsort_e:
            e_n_p.append(value[1])
            e_n_r.append(value[0])

        c_n_p = []
        c_n_r = []
        for value in listsort_c:
            c_n_p.append(value[1])
            c_n_r.append(value[0])

        fig = plt.figure()
        plt.title("P-R curve (Defect : {})".format(classes[i]))
        plt.axis([0.93, 1.01, 0.6, 1.01])

        plt.plot(r_n_r, r_n_p, label='Resnet')
        plt.plot(d_n_r, d_n_p, label='Densenet')
        plt.plot(e_n_r, e_n_p, label='Efficientnet')
        plt.plot(c_n_r, c_n_p, label='Capsnet')

        plt.grid(True)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend()

        plt.savefig('../p_r_data/p_r_curve/p-r_curve_{}.png'.format(classes[i]))
        plt.clf()