import argparse
import os

def dataset_info(dataset_name):
    dataset_info = dict()
    if  dataset_name == 'wm_811k':
        dataset_info['data_path'] = '/storage/mskim/WDM/datadistbalanced/test_aug/'
        dataset_info['data_height'] = 52
        dataset_info['data_width'] = 52
        dataset_info['batch_size'] = 40
        dataset_info['epochs'] = 20
        # dataset_info['network_name'] = 'lenet'
        # dataset_info['network_name'] = 'Alexnet'
        # dataset_info['network_name'] = 'VGGnet'
        # dataset_info['network_name'] = 'VGGnet_jh'
        # dataset_info['network_name'] = 'googlenet'
        # dataset_info['network_name'] = 'Resnet'
        # dataset_info['network_name'] = 'Resnet_18'
        # dataset_info['network_name'] = 'Resnet_nopadded'
        # dataset_info['network_name'] = 'Resnet_bottleneck'
        dataset_info['network_name'] = 'Densenet'
        # dataset_info['data_depth'] = 1

    elif  dataset_name == 'wm_811k_limited':
        dataset_info['data_path'] = '/storage/mskim/WDM/datalimited/test_aug/'
        dataset_info['data_height'] = 52
        dataset_info['data_width'] = 52
        dataset_info['batch_size'] = 40
        dataset_info['epochs'] = 20
        # dataset_info['network_name'] = 'lenet'
        # dataset_info['network_name'] = 'Alexnet'
        # dataset_info['network_name'] = 'VGGnet'
        # dataset_info['network_name'] = 'googlenet'
        # dataset_info['network_name'] = 'Resnet'
        # dataset_info['network_name'] = 'Resnet_18'
        # dataset_info['network_name'] = 'Resnet_nopadded'
        # dataset_info['network_name'] = 'Resnet_bottleneck'
        # dataset_info['network_name'] = 'Densenet'
        # dataset_info['data_depth'] = 1

    elif dataset_name == 'data52only':
        dataset_info['data_path'] = '/storage/hrlee/WDM/data52only/test_aug/'
        dataset_info['data_height'] = 52
        dataset_info['data_width'] = 52
        dataset_info['batch_size'] = 40
        dataset_info['epochs'] = 20
        # dataset_info['network_name'] = 'lenet'
        # dataset_info['network_name'] = 'Alexnet'
        # dataset_info['network_name'] = 'VGGnet'
        # dataset_info['network_name'] = 'VGGnet_jh'
        # dataset_info['network_name'] = 'googlenet'
        # dataset_info['network_name'] = 'Resnet'
        # dataset_info['network_name'] = 'Resnet_18'
        # dataset_info['network_name'] = 'Resnet_nopadded'
        # dataset_info['network_name'] = 'Resnet_bottleneck'
        # dataset_info['network_name'] = 'Densenet'
        # dataset_info['data_depth'] = 1

    elif dataset_name == 'mnist':
        dataset_info['data_path'] = '/storage/mskim/WDM/mnist/test/'
        dataset_info['data_height'] = 52
        dataset_info['data_width'] = 52
        dataset_info['batch_size'] = 40
        dataset_info['epochs'] = 20
        # dataset_info['network_name'] = 'lenet'
        # dataset_info['network_name'] = 'Alexnet'
        # dataset_info['network_name'] = 'VGGnet'
        dataset_info['network_name'] = 'VGGnet_jh_mnist'
        # dataset_info['network_name'] = 'VGGnet_jh'
        # dataset_info['network_name'] = 'googlenet'
        # dataset_info['network_name'] = 'Resnet'
        # dataset_info['network_name'] = 'Resnet_18'
        # dataset_info['network_name'] = 'Resnet_nopadded'
        # dataset_info['network_name'] = 'Resnet_bottleneck'
        # dataset_info['network_name'] = 'Densenet'
        # dataset_info['data_depth'] = 1
    else:
        ValueError('There is no dataset named {}'.format(dataset_name))
    return dataset_info


class Config:
    dataset_info = dataset_info(dataset_name='mnist')
    dataset_name = 'mnist'
    network_name = dataset_info['network_name']
    save_path = './checkpoints/pre_test_{}_{}'.format(dataset_name,network_name)
    map_path = './confusion_map/map_data/'
    train_test_save_path = './train_test/' + network_name

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--network_name', type=str, default=self.network_name)
        self.parser.add_argument('--continue_train', type=bool, default=True)
        self.parser.add_argument('--epochs', type=int, default=self.dataset_info['epochs'])
        self.parser.add_argument('--batch_size', type=int, default=self.dataset_info['batch_size'])
        #####
        self.parser.add_argument('--dataset_path', type=str, default=self.dataset_info['data_path'])
        self.parser.add_argument('--dataset_name', type=str, default=self.dataset_name)
        self.parser.add_argument('--data_height', type=str, default=self.dataset_info['data_height'])
        self.parser.add_argument('--data_width', type=str, default=self.dataset_info['data_width'])
        # self.parser.add_argument('--data_depth', type=str, default=self.dataset_info['data_depth'])
        #####
        self.parser.add_argument('--lr_policy', type=str, default='cosine', help='[step | plateau | cosine]')
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='monument for rmsprop optimizer')
        self.parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay')
        #####
        self.parser.add_argument('--save_path', type=str, default=self.save_path, help='path to store model')
        self.parser.add_argument('--train_test_save_path', type=str, default=self.train_test_save_path, help='')
        self.parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda')
        self.parser.add_argument('--gpu_id', type=str, default=0, help='gpu id used to train')
        self.parser.add_argument('--phase', type=str, default='train')
        self.parser.add_argument('--freq_show_loss', type=int, default=20)
        self.parser.add_argument('--freq_show_img', type=int, default=15)

        self.opt, _ = self.parser.parse_known_args()

    def print_options(self):
        """Print and save options
                It will print both current options and default values(if different).
                It will save options into a text file / [checkpoints_dir] / opt.txt
                """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(self.opt.save_path)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(self.opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)