import numpy as np
import torch
import time

def draw_chart(label, prediction):
    predict = prediction
    label = torch.argmax(label, dim=2)
    chart = np.zeros([label.shape[1] + 1, label.shape[1] + 1])
    # class의 수에 맞게 2차원의 빈 배열 생성, -- > row : label.shape[1], -- > col : label.shape[1]
    # row는 label, col은 prediction
    # normal이 경우에도 확률을 표시해줘야하므로 class의 수에 맞게 2차원의 빈 배열 생성시 행과 열에 +1을 해주어야함
    row = torch.argmax(label, dim=1)
    cnt = 0
    for batch in range(len(row)): # predict.shape[0]는 batch_size, 즉 batch_size만큼 반복
        col = torch.where(predict[batch])
        if torch.sum(predict[batch]) == 0: # predict가 normal인 경우
            if torch.sum(label[batch]) == 0:  # label도 normal경우 마지막 행에 count
                chart[-1, -1] += 1
            else:                           # label이 normal이 아닌경우 해당 행에 count
                chart[row[batch], -1] += 1

        else:                               # predict가 normal이 아닌 경우
            if torch.sum(label[batch]) == 0:  # label이 normal경우 마지막 행의 해당 열에 count
                for j in range(len(col[0])):
                    chart[-1, col[0][j]] += 1
            else:                           # label이 normal이 아닌경우 해당 행의 해당 열에 count
                for j in range(len(col[0])):
                    chart[row[batch], col[0][j]] += 1
        cnt += 1 # batch_size에 맞게 들어오는지 확인하기 위해 넣어줌 --> 사실상 필요없음
    return chart

def draw_chart_update(chart_update):
    map = np.zeros_like(chart_update)
    for k in range(map.shape[0]):
        if np.sum(chart_update[k]) !=0:
            map[k] = (chart_update[k]/np.sum(chart_update[k])) * 100
    np.set_printoptions(precision=2) # 소수점 둘째자리까지
    return map