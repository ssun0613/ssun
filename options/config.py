import argparse
import os

def dataset_info(dataset_name):
    dataset_info = dict()
    if  dataset_name == 'wm_811k':
        dataset_info['dataset_path_train'] = '/storage/mskim/WDM/datadistbalanced/train_aug/'
        dataset_info['dataset_path_test'] = '/storage/mskim/WDM/datadistbalanced/test_aug/'
        dataset_info['data_height'] = 52
        dataset_info['data_width'] = 52
        dataset_info['batch_size'] = 20
        # dataset_info['data_depth'] = 1

    elif  dataset_name == 'wm_811k_limited':
        dataset_info['dataset_path_train'] = '/storage/mskim/WDM/datalimited/train_aug/'
        dataset_info['dataset_path_test'] = '/storage/mskim/WDM/datalimited/test_aug/'
        dataset_info['data_height'] = 52
        dataset_info['data_width'] = 52
        dataset_info['batch_size'] = 20
        # dataset_info['data_depth'] = 1

    elif dataset_name == 'data52only':
        dataset_info['dataset_path_train'] = '/storage/hrlee/WDM/data52only/train_aug/'
        dataset_info['dataset_path_test'] = '/storage/hrlee/WDM/data52only/test_aug/'
        dataset_info['data_height'] = 224
        dataset_info['data_width'] = 224
        dataset_info['batch_size'] = 20
        # dataset_info['data_depth'] = 1

    elif dataset_name == 'multi':
        dataset_info['dataset_path_train'] = '/storage/hrlee/WDM/wdmmix_new/train_aug/'
        dataset_info['dataset_path_test'] = '/storage/hrlee/WDM/wdmmix_new/test_aug/'
        dataset_info['loss_name'] = 'mse'

        dataset_info['data_height'] = 52
        dataset_info['data_width'] = 52
        dataset_info['data_depth'] = 3
        dataset_info['batch_size'] = 40

        dataset_info['in_dim'] = 16
        dataset_info['out_dim'] = 24
        dataset_info['out_channels'] = 1024
        dataset_info['num_routing'] = 3
        dataset_info['threshold'] = 0.5


    else:
        ValueError('There is no dataset named {}'.format(dataset_name))
    return dataset_info


class Config:
    map_path = './confusion_map/map_data/'

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--network_name', type=str, default='capsnet') # [ resnet | densenet | efficientnet | capsnet ]
        self.parser.add_argument('--weight_name', type=str, default='capsnet') # [ resnet | densenet | efficientnet | capsnet ]
        self.parser.add_argument('--dataset_name', type=str, default='multi')
        self.parser.add_argument('--continue_train', type=bool, default=False)
        self.parser.add_argument('--epochs', type=int, default=20)
        #
        temp_parser, _ = self.parser.parse_known_args()
        self.dataset_info = dataset_info(dataset_name=temp_parser.dataset_name)
        #
        self.parser.add_argument('--batch_size', type=int, default=self.dataset_info['batch_size'])
        self.parser.add_argument('--dataset_path_train', type=str, default=self.dataset_info['dataset_path_train'])
        self.parser.add_argument('--dataset_path_test', type=str, default=self.dataset_info['dataset_path_test'])
        self.parser.add_argument('--loss_name', type=str, default=self.dataset_info['loss_name'])

        self.parser.add_argument('--data_height', type=int, default=self.dataset_info['data_height'])
        self.parser.add_argument('--data_width', type=int, default=self.dataset_info['data_width'])
        self.parser.add_argument('--data_depth', type=int, default=self.dataset_info['data_depth'])

        self.parser.add_argument('--in_dim', type=int, default=self.dataset_info['in_dim'])
        self.parser.add_argument('--out_dim', type=int, default=self.dataset_info['out_dim'])
        self.parser.add_argument('--out_channels', type=int, default=self.dataset_info['out_channels'])
        self.parser.add_argument('--num_routing', type=int, default=self.dataset_info['num_routing'])
        self.parser.add_argument('--threshold', type=float, default=self.dataset_info['threshold'])
        #####
        self.parser.add_argument('--scheduler_name', type=str, default='cosine', help='[stepLR | cycliclr | cosine]')
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--optimizer_name', type=str, default='Adam', help='[Adam | RMSprop]')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='monument for rmsprop optimizer')
        self.parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay')
        #####
        self.parser.add_argument('--save_path', type=str, default='./checkpoints/pre_test_{}_{}'.format(temp_parser.dataset_name, temp_parser.network_name), help='path to store model')
        self.parser.add_argument('--train_test_save_path', type=str, default='./train_test/' + temp_parser.network_name, help='')
        self.parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda')
        self.parser.add_argument('--gpu_id', type=str, default='0', help='gpu id used to train')
        self.parser.add_argument('--phase', type=str, default='train')
        self.parser.add_argument('--freq_show_loss', type=int, default=100)
        self.parser.add_argument('--freq_show_image', type=int, default=200)
        self.parser.add_argument('--freq_save_net', type=int, default=50)
        self.parser.add_argument('--num_test_iter', type=int, default=5)

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
