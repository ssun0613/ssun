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
import torch.nn.functional as F
import time
from torch.optim import lr_scheduler

from options.config import Config
from options.draw_chart import draw_chart
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
    if config.opt.network_name == None:
        ValueError('Please set model_name!')

    elif config.opt.network_name == 'efficientnet':
        from model.efficientnet import efficientnet_b0 as network
        net = network().to(device)

    elif config.opt.network_name == 'efficientnet_GAIN':
        from model.efficientnet import efficientnet_gain_b0 as network
        net = network().to(device)

    elif config.opt.network_name == 'capsnet':
        from model.capsulenet import capsnet as network
        net = network().to(device)

    if not config.opt.continue_train:
        net.init_weights()
    return net

def calc_loss(opt, net, input_label, input_image=None):
    if 'GAIN' in opt.network_name:
        from model.loss import multiloss
        cl_out = net.get_outputs()
        am_out = net.get_outputs_2()

        fn_loss_1 = multiloss(input_label, cl_out)

        exist_label = input_label[:, :, 1]
        am_out_score = F.softmax(am_out, dim=2)[:, :, 1]
        loss_cnt = 0
        fn_loss_2 = 0
        for b_id in range(exist_label.shape[0]):
            if len(torch.where(exist_label[b_id])[0]) != 0:
                loss_cnt += 1
                fn_loss_2 += am_out_score[b_id, torch.where(exist_label[b_id])].sum()
            else:
                loss_cnt += 1
                fn_loss_2 += am_out_score[b_id].sum()
        fn_loss_2 = fn_loss_2 / loss_cnt

        if np.isnan(fn_loss_2.cpu().detach().numpy()):
            print('fn_loss_2 is NaN....')
        fn_loss = fn_loss_1 + fn_loss_2

    else:
        from model.loss import multiloss
        output = net.get_outputs()
        fn_loss = multiloss(input_label, output)

    return fn_loss

def setup_dataset(opt):
    if opt.dataset_name in ['wm_811k', 'wm_811k_limited']:
        from dataset.wm_811k_45000 import WM_811K
        dataset_object_train = WM_811K(dataset_path=config.opt.dataset_path_train, is_training=True,
                                       data_size=(config.opt.data_height, config.opt.data_width))
        train_loader = torch.utils.data.DataLoader(dataset_object_train,
                                                   batch_size=config.opt.batch_size,
                                                   shuffle=True, num_workers=0)
        dataset_object_test = WM_811K(dataset_path=config.opt.dataset_path_test, is_training=True,
                                      data_size=(config.opt.data_height, config.opt.data_width))
        test_loader = torch.utils.data.DataLoader(dataset_object_test,
                                                  batch_size=config.opt.batch_size,
                                                  shuffle=True, num_workers=0)
    elif opt.dataset_name in ['multi']:
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
        NotImplementedError('Not implemented {}'.format(opt.dataset_name))
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
    temp_acc = []
    chart_update = np.zeros([9, 9])
    test_iter = iter(test_loader)

    for curr_epoch in range(config.opt.epochs):
        epoch_start_time = time.time()
        print('------- epoch {} starts -------'.format(curr_epoch + 1))
        for batch_id, data in enumerate(train_loader, 1):
            global_step += 1

            input_image = data['image'].type('torch.FloatTensor').to(device)
            input_label = data['label'].type('torch.FloatTensor').to(device)

            label_1 = input_label.unsqueeze(dim=2)
            label_2 = torch.ones_like(label_1)

            input_label = torch.concat([(label_2 - label_1), label_1], dim=2)

            if 'GAIN' in config.opt.network_name:
                net.set_input(input_image, input_label)

            else:
                net.set_input(input_image)

            net.forward()
            fn_loss = calc_loss(config.opt, net, input_label)

            optimizer.zero_grad()
            fn_loss.backward()
            optimizer.step()

            predict = net.predict()
            prediction = torch.argmax(predict, dim=2)
            accuracy = net.accuracy(prediction, input_label)

            temp_loss.append(fn_loss.cpu().detach().numpy().item())
            temp_acc.append(accuracy)

            if global_step % config.opt.freq_show_loss == 0:
                loss_dict = dict()
                loss_dict['epoch'] = curr_epoch
                loss_dict['loss'] = np.mean(temp_loss)
                loss_dict['acc'] = np.mean(temp_acc)
                wb_logger.log(prefix='train', metrics_dict=loss_dict)

                temp_loss = []
                temp_acc = []

            # if global_step % config.opt.freq_show_heatmap == 0:
            #     if config.opt.show_cam != None:
            #         if 'GAIN' in config.opt.network_name:
            #             from model.CAM import GradCAM, Gain_GradCAM
            #             GradCAM, input_image_pick, input_attention__pick, input_make_pick, input_label_pick, prediction_pick = Gain_GradCAM(net, input_label)
            #             wb_logger.log_images_to_wandb_multi(input_image_pick, GradCAM, mode='GAIN_GradCAM' + "_" + str(config.opt.dataset_name) + "{}".format(global_step))
            #             print("check wandb\n")
            #
            #         else:
            #             from model.CAM import GradCAM
            #             GradCAM, input_image_pick, input_label_pick, prediction_pick = GradCAM(net, input_label)
            #             wb_logger.log_images_to_wandb_multi(input_image_pick, GradCAM, mode='train_GradCAM' + "_" + str(config.opt.dataset_name) + "{}".format(global_step))
            #             print("check wandb\n")
            #
            #     else:
            #         print("Don't want to show the CAM or GradCAM")
            #     net.train()

            if global_step % config.opt.freq_save_net == 0:
                torch.save({'net': net.state_dict()},
                           config.opt.save_path + '/{}_epoch'.format(config.opt.network_name) + ".pth")

        print('Elapsed time for one epoch: %.2f [s]' % (time.time() - epoch_start_time))
        print('------- epoch {} ends -------'.format(curr_epoch + 1))

        print('Update Learning Rate...')
        scheduler.step()
        print('[Network] learning rate = %.7f' % optimizer.param_groups[0]['lr'])