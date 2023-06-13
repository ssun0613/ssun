import os
os.environ["CONFIG_PATHS"] = '/storage/mskim/wandb/'
os.environ["WANDB_DIR"] = '/storage/mskim/wandb/'
os.environ["WANDB_CACHE_DIR"] = '/storage/mskim/wandb/'
os.environ["WANDB_CONFIG_DIR"] = '/storage/mskim/wandb/'
os.environ["WANDB_RUN_DIR"] = '/storage/mskim/wandb/'

import sys
sys.path.append("..")
import timeit
import torch.nn as nn
from torch.optim import lr_scheduler
from options.config import Config
from options.precision_recall import *

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

    elif opt.network_name == 'lenet':
        from model.lenet import LeNet as network
        net = network(opt).to(device)

    elif opt.network_name == 'resnet':
        from model.Resnet import Resnet as network
        net = network(opt).to(device)

    elif opt.network_name == 'resnet_2':
        from model.Resnet_2 import Resnet as network
        net = network(opt).to(device)

    elif opt.network_name == 'densenet':
        from model.DenseNet import DenseNet as network
        net = network(opt).to(device)

    elif opt.network_name == 'efficientnet':
        from model.efficientnet import EfficientNet as network
        net = network(opt).to(device)

    elif opt.network_name == 'capsnet':
        from model.capsulenet import capsnet as network
        net = network(opt).to(device)

    if not opt.continue_train:
        net.init_weights()
    else:
        net = net.load_networks(net=net, loss_type=opt.loss_name, weight_path=opt.save_path + '/', device=device)
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

    train_loader, test_loader = setup_dataset(config.opt)

    net = setup_network(config.opt, device)

    test_index = 0
    t_p = []
    t_r = []

    mean_time = 0

    for threshold in range(1, 101):

        TP = np.zeros([9, 1])
        FP = np.zeros([9, 1])
        FN = np.zeros([9, 1])

        config.opt.threshold = round(threshold * 0.01, 5)

        elapsed_time = 0
        for batch_id, data in enumerate(test_loader, 1):

            with torch.no_grad():

                input_image = data['image'].type('torch.FloatTensor').to(device)
                input_label = data['label'].type('torch.FloatTensor').to(device)

                net.set_input(input_image)

                start_time = timeit.default_timer()

                if config.opt.network_name == 'capsnet':
                  net.forward(mode='test')
                else:
                  net.forward()
                predict = net.predict()

                end_time = timeit.default_timer()
                elapsed_time += end_time - start_time

                TP, FP, FN = calc_precision_recall_1(predict.cpu(), input_label.cpu(), TP, FP, FN)
                precision, recall = precision_recall(TP, FP, FN)

        mean_time += elapsed_time
        t_p.append(precision.reshape(1,-1))
        t_r.append(recall.reshape(1,-1))
    np.save('./p_r_data/p_r_data_{}/p_r_data_t_p_{}_{}_{}_{}_101_conv'.format(config.opt.network_name, config.opt.network_name, config.opt.loss_name, config.opt.in_dim, config.opt.out_channels), t_p)
    np.save('./p_r_data/p_r_data_{}/p_r_data_t_r_{}_{}_{}_{}_101_conv'.format(config.opt.network_name, config.opt.network_name, config.opt.loss_name, config.opt.in_dim, config.opt.out_channels), t_r)
    # np.savez('./p_r_data/p_r_data_{}/p_r_data_{}_{}_{}_{}_1000.npz'.format(config.opt.network_name, config.opt.network_name, config.opt.loss_name, config.opt.in_dim, config.opt.out_channels), x=t_p, y=t_r)

    print("%f [sec]" % (mean_time))
    print('finish')

    # TP = np.zeros([9, 1])
    # FP = np.zeros([9, 1])
    # FN = np.zeros([9, 1])
    #
    # config.opt.threshold = 0.5
    # cnt = 0
    # elapsed_time = 0
    # for batch_id, data in enumerate(test_loader, 1):
    #     cnt += 1
    #     if cnt > 4000:
    #         break
    #     with torch.no_grad():
    #
    #         input_image = data['image'].type('torch.FloatTensor').to(device)
    #         input_label = data['label'].type('torch.FloatTensor').to(device)
    #
    #         net.set_input(input_image)
    #         #
    #         start_time = timeit.default_timer()
    #
    #         if config.opt.network_name == 'capsnet':
    #             net.forward(mode='test')
    #         else:
    #             net.forward()
    #         predict = net.predict()
    #
    #         end_time = timeit.default_timer()
    #         elapsed_time += end_time - start_time
    #
    #         TP, FP, FN = calc_precision_recall_1(predict.cpu(), input_label.cpu(), TP, FP, FN)
    #         precision, recall = precision_recall(TP, FP, FN)
    #
    # print("%f [sec]" % elapsed_time)
    #
    # t_p.append(precision.reshape(1,-1))
    # t_r.append(recall.reshape(1,-1))
    # # np.save('./p_r_data/p_r_data_{}/p_r_data_t_p_{}_{}_{}_{}'.format(config.opt.network_name, config.opt.network_name, config.opt.loss_name, config.opt.in_dim, config.opt.out_channels), t_p)
    # # np.save('./p_r_data/p_r_data_{}/p_r_data_t_r_{}_{}_{}_{}'.format(config.opt.network_name, config.opt.network_name, config.opt.loss_name, config.opt.in_dim, config.opt.out_channels), t_r)
    # np.savez('./p_r_data/p_r_data_{}/p_r_data_{}_{}_{}_{}_2.npz'.format(config.opt.network_name, config.opt.network_name, config.opt.loss_name, config.opt.in_dim, config.opt.out_channels), x=t_p, y=t_r)
    # print('finish')

