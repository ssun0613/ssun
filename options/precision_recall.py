import numpy as np
import torch
import torch.nn.functional as F

def calc_precision_recall(prediction, label, TP, FP, FN):
    for batch_size in range(prediction.shape[0]):

        prediction_check = torch.where(prediction[batch_size] == 1)[0]
        label_check = torch.where(label[batch_size] == 1)[0]

        if len(label_check) > 0:
            t = torch.eq(prediction_check, label_check)
            f = ~t

            TP[prediction_check[t]] +=1
            FP[prediction_check[f]] +=1
            FN[label_check[f]] +=1

        elif len(label_check)==0:

            if len(prediction_check)==0:
                TP[8] +=1

            elif len(prediction_check)!=0:
                FN[8] +=1

    return TP, FP, FN


def calc_precision_recall_1(prediction, label, TP, FP, FN):
    for batch_size in range(prediction.shape[0]):
        prediction_check = torch.where(prediction[batch_size] == 1)[0]
        label_check = torch.where(label[batch_size] == 1)[0]

        t_p = []
        t_l = []

        if len(prediction_check) > 0 and len(label_check) > 0:
            for i in range(len(prediction_check)):
                a = torch.eq(prediction_check[i], label_check)
                t_l.append(a)
            t_l = torch.stack(t_l, dim=0)
            t_l = torch.sum(t_l, dim=0, dtype=bool)

            for j in range(len(label_check)):
                b = torch.eq(prediction_check, label_check[j])
                t_p.append(b)
            t_p = torch.stack(t_p, dim=0)
            t_p = torch.sum(t_p, dim=0, dtype=bool)

            f_p = ~t_p
            f_l = ~t_l

            TP[prediction_check[t_p]] +=1
            FP[prediction_check[f_p]] +=1
            FN[label_check[f_l]] +=1

        elif len(prediction_check) == 0 and len(label_check) == 0:
            TP[8] += 1

        elif len(prediction_check) == 0 and len(label_check) != 0:
            FN[label_check] += 1
            FP[8] += 1

        elif len(prediction_check) != 0 and len(label_check) == 0:
            FP[prediction_check] += 1
            FN[8] += 1


    return TP, FP, FN

def precision_recall(TP, FP, FN):
    recall = np.zeros([9, 1])
    precision = np.zeros([9, 1])
    #              TP
    # Recall = ---------
    #           TP + FN
    for i in range(recall.shape[0]):
        if (TP[i] + FN[i]) != 0:
            temp = TP[i] / (TP[i] + FN[i])
        else:
            temp = 0
        recall[i] = temp
    #                TP
    # Precision = ---------
    #              TP + FP
    for i in range(precision.shape[0]):
        if (TP[i] + FP[i]) != 0:
            temp = TP[i] / (TP[i] + FP[i])
        else:
            temp = 0
        precision[i] = temp
    return precision, recall

if __name__ == '__main__':
    TP = np.zeros([9, 1])
    FP = np.zeros([9, 1])
    FN = np.zeros([9, 1])


    prediction = torch.tensor([[0, 0, 1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 0, 1, 0, 0]])
    label = torch.tensor([[0, 1, 1, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0]])

    TP_2, FP_2, FN_2 = calc_precision_recall_1(prediction, label, TP, FP, FN)
    precision, recall = precision_recall(TP_2, FP_2, FN_2)

    print(TP_2, FP_2, FN_2)