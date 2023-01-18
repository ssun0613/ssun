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

        if len(label_check) > 0:

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

        elif len(label_check) == 0:
            if len(prediction_check) == 0:
                TP[8] += 1
            elif len(prediction_check) != 0:
                FN[8] += 1

    return TP, FP, FN

if __name__ == '__main__':
    TP = np.zeros([9, 1])
    FP = np.zeros([9, 1])
    FN = np.zeros([9, 1])

    # prediction = torch.tensor([[0, 1, 1, 0, 0, 1, 0, 0]])
    # label = torch.tensor([[1, 0, 1, 0, 0, 1, 0, 0]])

    # prediction = torch.tensor([[0, 1, 1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 0, 0]])
    # label = torch.tensor([[1, 0, 1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0]])

    prediction = torch.tensor([[0, 0, 1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 0, 1, 0, 0]])
    label = torch.tensor([[0, 1, 1, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0]])

    # TP_1, FP_1, FN_1 = calc_precision_recall(prediction, label, TP, FP, FN)
    TP_2, FP_2, FN_2 = calc_precision_recall_1(prediction, label, TP, FP, FN)

    print(TP_1, FP_1, FN_1)
    print(TP_2, FP_2, FN_2)