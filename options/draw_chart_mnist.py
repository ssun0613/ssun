import numpy as np
import torch
import time

def draw_chart(label, prediction):
    col = prediction
    row = torch.argmax(label, dim=1) # input_label -->
    chart = np.zeros([label.shape[1], label.shape[1]])
    cnt = 0
    for batch in range(row.shape[0]):
        chart[col[batch], row[batch]] += 1
    return chart

def draw_chart_update(chart_update):
    map = np.zeros_like(chart_update)
    for k in range(map.shape[0]):
        if np.sum(chart_update[k]) !=0:
            map[k] = chart_update[k]/np.sum(chart_update[k])
    np.set_printoptions(precision=2) # 소수점 둘째자리까지
    return map