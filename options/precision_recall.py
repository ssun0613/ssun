import numpy as np
import torch
import torch.nn.functional as F

def calc_precision_recall(prediction, label, TP, FP, FN):
    for idx in range(label.shape[0]):
        prediction_check = torch.where(prediction[idx] == 1)[0]
        label_check = torch.where(label[idx] == 1)[0]

        if len(prediction_check) > 0:
            index_check = (prediction[idx] == label[idx])





if __name__ == '__main__':
    TP = np.zeros([9, 1])
    FP = np.zeros([9, 1])
    FN = np.zeros([9, 1])

    prediction = torch.tensor([[0, 0, 1, 0, 0, 1, 0, 0]])
    label = torch.tensor([[1, 0, 1, 0, 0, 1, 0, 0]])

    calc_precision_recall(prediction, label, TP, FP, FN)