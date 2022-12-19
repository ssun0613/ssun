import torch
import datetime
import os
import numpy as np
import wandb

from utils import common

class WBLogger_1:
    def __init__(self, opts):
        self.batch_size = opts.batch_size
        # wandb.init(project="change", config=vars(opts))
        wandb.init(project=opts.dataset_name, config=vars(opts))

    @staticmethod
    def log(prefix, metrics_dict):
        log_dict = {f'{prefix}_{key}': value for key, value in metrics_dict.items()}
        wandb.log(log_dict)

    @staticmethod
    def log_images_to_wandb(x, y, y_hat, id_logs, prefix, step, opts):
        im_data = []
        column_names = ["Source", "Target", "Output"]
        if id_logs is not None:
            column_names.append("ID Diff Output to Target")
        for i in range(len(x)):
            cur_im_data = [
                wandb.Image(common.log_input_image(x[i], opts)),
                wandb.Image(common.tensor2im(y[i])),
                wandb.Image(common.tensor2im(y_hat[i])),
            ]
            if id_logs is not None:
                cur_im_data.append(id_logs[i]["diff_target"])
            im_data.append(cur_im_data)
        outputs_table = wandb.Table(data=im_data, columns=column_names)
        wandb.log({f"{prefix.title()} Step {step} Output Samples": outputs_table})

    @staticmethod
    # def log_images_to_wandb_2(prefix, image, label, output, step):
    #     im_data = []
    #     column_names = ["image", "label", "output"]
    #     for i in range(image.shape[0]):
    #         cur_im_data = [
    #             wandb.Image(common.tensor2numpy(image[i])),
    #             wandb.Image(common.tensor2numpy(label[i])),
    #             wandb.Image(common.tensor2numpy(output[i])),
    #         ]
    #         im_data.append(cur_im_data)
    #     outputs_table = wandb.Table(data=im_data, columns=column_names)
    #     wandb.log({f"{prefix.title()} Step {step} Output Samples": outputs_table})

    def log_images_to_wandb_2(prefix, image, prediction, step):
        im_data = []
        column_names = ["image"]
        for i in range(image.shape[0]):
            cur_im_data = [
                wandb.Image(common.tensor2numpy(image[i]))
            ]
            im_data.append(cur_im_data)
        outputs_table = wandb.Table(data=im_data, columns=column_names)
        wandb.log({f"{prefix.title()} Step {step} Output Samples": outputs_table})


    def log_confusionMap(self, map, network_name, dataset_name, curr_epoch, global_step, mode):
        # logging_title = 'Heatmap_' + mode if mode is not None else 'Heatmap'
        x_label = ['C', 'D', 'EL', 'ER', 'L', 'NF', 'R', 'S', 'N']
        y_label = ['C', 'D', 'EL', 'ER', 'L', 'NF', 'R', 'S', 'N']
        # x_label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        # y_label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        wandb.log({ mode + 'Heatmap_'+str(network_name) +'_' +str(dataset_name) +'_' +str(curr_epoch) +'_' +str(global_step) : wandb.plots.HeatMap(x_label, y_label, map, show_text=True)})
