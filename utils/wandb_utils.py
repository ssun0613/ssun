import torch
import datetime
import os
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from utils import common

class WBLogger:
    def __init__(self, opts):
        self.batch_size = opts.batch_size
        wandb.init(project=opts.network_name, config=vars(opts))

    @staticmethod
    def log(prefix, metrics_dict):
        log_dict = {f'{prefix}_{key}': value for key, value in metrics_dict.items()}
        wandb.log(log_dict)

    @staticmethod
    def log_images_to_wandb_1(image, label, prediction):
        data = []
        label = torch.argmax(torch.argmax(label, dim=2), dim=1)
        prediction = torch.argmax(prediction, dim=1)

        column_names = ["image", "label", "predict"]

        for i in range(image.shape[0]):
            data_new = [
                wandb.Image(common.tensor2numpy(image[i])),
                np.array2string(label[i].cpu().numpy()),
                np.array2string(prediction[i].cpu().numpy())
            ]
            data.append(data_new)
        outputs_table = wandb.Table(data=data, columns=column_names)
        wandb.log({"train_test": outputs_table})

    def log_images_to_wandb_2(self, image, input_label, prediction, mode):
        logging_title = mode
        data = []
        input_label = torch.argmax(input_label, dim=2)
        classes = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Local", "Near_full", "Scratch", "Random"]
        column_names = ["image", "label", "predict"]

        for i in range(len(image)):
            prediction_1 = []
            image_1 = common.tensor2numpy(image[i])
            if len(torch.where(prediction[i])[0]) == 1 :
                if torch.sum(input_label[i]) == 0:
                    label_1 = "Normal"
                    if torch.sum(prediction[i]) == 0:
                        prediction_1.append("Normal")
                    else:
                        prediction_1.append(classes[torch.argmax(prediction[i])])
                else:
                    label_1 = classes[torch.argmax(input_label[i])]
                    if torch.sum(prediction[i]) == 0:
                        prediction_1.append("Normal")
                    else:
                        prediction_1.append(classes[torch.argmax(prediction[i])])

            elif len(torch.where(prediction[i])[0]) > 1 :
                for j in range(len(torch.where(prediction[i])[0])):
                    if torch.sum(input_label[i]) == 0:
                        label_1 = "Normal"
                        prediction_1.append(classes[torch.where(prediction[i])[0][j]])

                    else:
                        label_1 = classes[torch.argmax(input_label[i])]
                        prediction_1.append(classes[torch.where(prediction[i])[0][j]])

            data_new = [wandb.Image(image_1),
                        label_1,
                        ' + '.join(prediction_1)]
            data.append(data_new)
        outputs_table = wandb.Table(data=data, columns=column_names)
        wandb.log({logging_title: outputs_table})

    def log_images_to_wandb_multi(self, image, GradCAM,mode):
        logging_title = mode
        data = []
        column_names = ["image", "GradCAM"]

        for i in range(len(image)):
            image_1 = common.tensor2numpy(image[i])
            GradCAM = common.tensor2numpy(GradCAM[i])
            data_new = [wandb.Image(image_1),
                        wandb.Image(GradCAM)]
            data.append(data_new)
        outputs_table = wandb.Table(data=data, columns=column_names)
        wandb.log({logging_title: outputs_table})
