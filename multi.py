import os
os.environ["CONFIG_PATHS"] = '/storage/mskim/wandb/'
os.environ["WANDB_DIR"] = '/storage/mskim/wandb/'
os.environ["WANDB_CACHE_DIR"] = '/storage/mskim/wandb/'
os.environ["WANDB_CONFIG_DIR"] = '/storage/mskim/wandb/'
os.environ["WANDB_RUN_DIR"] = '/storage/mskim/wandb/'

import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.optim import lr_scheduler

from options.config import Config
from options.precision_recall import calc_precision_recall_1, precision_recall
from utils.wandb_utils import WBLogger

def setup_scheduler(opt, optimizer):
    if opt.scheduler_name == 'steplr':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5)
    elif opt.scheduler_name == 'cycliclr':
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-9, max_lr=opt.lr, cycle_momentum=False, step_size_up=3, step_size_down=17, mode='triangular2')
    elif opt.scheduler_name == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-9)
    else:
        scheduler = None
        NotImplementedError('{} not implemented'.format(opt.scheduler_name))
    return scheduler

def setup_optimizer(opt, net):
    if opt.optimizer_name == 'Adam':
        optim = torch.optim.Adam(net.parameters(), lr=config.opt.lr)
    elif opt.optimizer_name == 'RMSprop':
        optim = torch.optim.RMSprop(net.parameters(), lr=config.opt.lr)
    else:
        optim = None
        NotImplementedError('{} not implemented'.format(opt.optimizer_name))
    return optim

def setup_network(opt, device):
    if opt.network_name == None:
        ValueError('Please set model_name!')

    elif opt.network_name == 'resnet':
        from model.Resnet import Resnet as network
        net = network(opt).to(device)

    elif opt.network_name == 'densenet':
        from model.DenseNet import DenseNet as network
        net = network(opt).to(device)

    elif opt.network_name == 'efficientnet':
        from model.efficientnet import efficientnet_b0 as network
        net = network(opt).to(device)

    elif opt.network_name == 'capsnet':
        from model.capsulenet_1 import capsnet as network
        net = network(opt).to(device)

    if not opt.continue_train:
        net.init_weights()
    else:
        net = net.load_networks(net=net, net_type=opt.weight_name, weight_path=opt.save_path + '/pre_tested', device=device)
    return net

def calc_loss(net, input_label):
    if config.opt.loss_name == 'mse':
        fn_loss = nn.MSELoss()
        loss = fn_loss(net.get_output(), input_label)

    elif config.opt.loss_name == 'cross':
        fn_loss = nn.CrossEntropyLoss()
        loss = fn_loss(net.get_output(), input_label)

    return loss

def setup_dataset(opt):
    if config.opt.dataset_name in ['multi']:
        from dataset.single_type_36000_mat import single_mat
        dataset_object_train = single_mat(dataset_path=config.opt.dataset_path_train,
                                          data_size=(config.opt.data_height, config.opt.data_width))
        train_loader = torch.utils.data.DataLoader(dataset_object_train,
                                                   batch_size=config.opt.batch_size,
                                                   shuffle=True, num_workers=0)
        dataset_object_test = single_mat(dataset_path=config.opt.dataset_path_test,
                                         data_size=(config.opt.data_height, config.opt.data_width))
        test_loader = torch.utils.data.DataLoader(dataset_object_test,
                                                  batch_size=config.opt.batch_size,
                                                  shuffle=True, num_workers=0)
    else:
        train_loader, test_loader = None, None
        NotImplementedError('Not implemented {}'.format(config.opt.dataset_name))
    return train_loader, test_loader

def setup_device(opt):
    if len(opt.gpu_id) != 0:
        device = torch.device("cuda:{}".format(config.opt.gpu_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device

if __name__ == '__main__':
    config = Config()
    config.print_options()
    device = setup_device(config.opt)
    wb_logger = WBLogger(config.opt)

    train_loader, test_loader = setup_dataset(config.opt)

    net = setup_network(config.opt, device)
    optimizer = setup_optimizer(config.opt, net)
    scheduler = setup_scheduler(config.opt, optimizer)

    global_step = 0
    temp_loss = []
    temp_precision = []
    temp_recall = []
    TP = np.zeros([9, 1])
    FP = np.zeros([9, 1])
    FN = np.zeros([9, 1])
    test_iter = iter(test_loader)

    for curr_epoch in range(config.opt.epochs):
        epoch_start_time = time.time()
        print('------- epoch {} starts -------'.format(curr_epoch + 1))
        for batch_id, data in enumerate(train_loader, 1):
            global_step += 1

            input_image = data['image'].type('torch.FloatTensor').to(device)
            input_label = data['label'].type('torch.FloatTensor').to(device)

            net.set_input(input_image)

            net.forward()

            fn_loss = calc_loss(net, input_label)

            optimizer.zero_grad()
            fn_loss.backward()
            optimizer.step()

            predict = net.predict()

            TP, FP, FN = calc_precision_recall_1(predict.cpu(), input_label.cpu(), TP, FP, FN)
            precision, recall = precision_recall(TP, FP, FN)

            temp_precision.append(precision)
            temp_recall.append(recall)

            temp_loss.append(fn_loss.cpu().detach().numpy().item())

            if global_step % 50 == 0:
                wb_logger.log_precision_recall_curve(temp_precision, temp_recall, 'train')

            if global_step % config.opt.freq_show_loss == 0:
                loss_dict = dict()
                loss_dict['epoch'] = curr_epoch
                loss_dict['loss'] = np.mean(temp_loss)
                wb_logger.log(prefix='train', metrics_dict=loss_dict)
                # precision, recall = precision_recall(TP, FP, FN)
                wb_logger.log_precision_recall(recall, precision, 'train')

                temp_loss = []
                temp_acc = []
                TP = np.zeros([9, 1])
                FP = np.zeros([9, 1])
                FN = np.zeros([9, 1])

            # if global_step % config.opt.freq_save_net == 0:
            #     torch.save({'net': net.state_dict()},
            #                config.opt.save_path + '/{}_epoch'.format(config.opt.network_name) + ".pth")

        print('Elapsed time for one epoch: %.2f [s]' % (time.time() - epoch_start_time))
        print('------- epoch {} ends -------'.format(curr_epoch + 1))

        print('Update Learning Rate...')
        scheduler.step()
        print('[Network] learning rate = %.7f' % optimizer.param_groups[0]['lr'])