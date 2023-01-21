import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import glob
# from WDM_SSUN.options.config_test import Config
from WDM_SSUN.options.config_laptop_test import Config

if __name__=='__main__':
    config = Config()
    map_data = sorted(glob.glob('./map_data/*npz'))

    # x_label = ['C', 'D', 'EL', 'ER', 'L', 'NF', 'R', 'S', 'N']
    # y_label = ['C', 'D', 'EL', 'ER', 'L', 'NF', 'R', 'S', 'N']
    #
    # map_1 = pd.DataFrame(np.load(map_data[2])['arr_0'], x_label, y_label)  # lenet
    # map_2 = pd.DataFrame(np.load(map_data[0])['arr_0'], x_label, y_label)  # Alexnet
    # map_3 = pd.DataFrame(np.load(map_data[1])['arr_0'], x_label, y_label)  # googleenet

    x_label = ['C', 'D', 'EL', 'ER', 'L', 'NF', 'R', 'S', 'N']
    y_label = ['C', 'D', 'EL', 'ER', 'L', 'NF', 'R', 'S', 'N']

    map_1 = pd.DataFrame(np.load(map_data[3])['arr_0'], x_label, y_label)  # lenet
    map_2 = pd.DataFrame(np.load(map_data[0])['arr_0'], x_label, y_label)  # Alexnet
    map_3 = pd.DataFrame(np.load(map_data[1])['arr_0'], x_label, y_label) # VGGnet
    map_4 = pd.DataFrame(np.load(map_data[2])['arr_0'], x_label, y_label) # googlenet

    fig, axs = plt.subplots(2, 2,figsize=(16, 16))
    fig.subplots_adjust(top=0.92, bottom=0.05, right=0.995, left=0.05, wspace=0.12 ,hspace=0.15)
    fig.suptitle("confusion_map", fontsize=24)

    sns.heatmap(map_1, annot=True, fmt='.2f', cmap='Blues', ax=axs[0][0])
    axs[0][0].set_xlabel("label")
    axs[0][0].set_ylabel("prediction")
    axs[0][0].set_title("Resnet", fontsize=16)

    sns.heatmap(map_2, annot=True, fmt='.2f', cmap='Blues', ax=axs[0][1])
    axs[0][1].set_xlabel("label")
    axs[0][1].set_ylabel("prediction")
    axs[0][1].set_title("Resnet", fontsize=16)

    sns.heatmap(map_3, annot=True, fmt='.2f', cmap='Blues', ax=axs[1][0])
    axs[1][0].set_xlabel("label")
    axs[1][0].set_ylabel("prediction")
    axs[1][0].set_title("Resnet", fontsize=16)

    sns.heatmap(map_4, annot=True, fmt='.2f', cmap='Blues', ax=axs[1][1])
    axs[1][1].set_xlabel("label")
    axs[1][1].set_ylabel("prediction")
    axs[1][1].set_title("Resnet", fontsize=16)

    # plt.show()
    # plt.savefig("/Users/ssun/Desktop/{}_{}.png".format(config.dataset_name,config.network_name))
    plt.savefig("/Users/ssun/Desktop/total.png")


