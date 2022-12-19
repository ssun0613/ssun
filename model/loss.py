import numpy as np
import torch
import torch.nn as nn
import time

def multiloss(input_label, output):

    fn_loss = nn.CrossEntropyLoss() #loss fcn로 CEE사용
    total_loss = 0
    loss = 0
    for i in range(input_label.shape[0]):  # batch_size만큼 반복
        pos_cnt = 0
        pos_loss = 0
        neg_cnt = 0
        neg_loss = 0
        for j in range(input_label.shape[1]):  # class 수만큼 반복
            if torch.argmax(input_label[i][j]) == 1: # class의 값이 1인 경우 즉,결함이 존재 하는 경우
                pos_cnt += 1
                pos_loss += fn_loss(output[i][j].unsqueeze(dim=0), input_label[i][j].unsqueeze(dim=0))
            elif torch.argmax(input_label[i][j]) == 0: # class의 값이 0인 경우 즉,결함이 존재 하지 않는 경우
                neg_cnt += 1
                neg_loss += fn_loss(output[i][j].unsqueeze(dim=0), input_label[i][j].unsqueeze(dim=0))
        if pos_cnt == 0:  # 학습 도중 pos_cnt가 0인 경우가 존재 할 수 있으므로 예외 처리 필요함
            loss += (neg_loss / neg_cnt)
        elif neg_cnt == 0:  # 학습 도중 neg_cnt가 0인 경우가 존재 할 수 있으므로 예외 처리 필요함
            loss += (pos_loss / pos_cnt)
        else:
            loss += (pos_loss / pos_cnt) + (neg_loss / neg_cnt)  # batch_size 만큼 저장

    total_loss = loss / int(input_label.shape[0])
    return total_loss

def singleloss(input_label, output):

    fn_loss = nn.CrossEntropyLoss() #loss fcn로 CEE사용
    loss = fn_loss(output, input_label)
    return loss