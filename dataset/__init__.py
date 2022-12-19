import torch

from torch.utils.data import DataLoader
from dataset.wm_811k_45000 import WM_811K

def create_dataloader(config):
    if  config.opt.dataset_name == 'wm_811k':
        dataset_object_train = WM_811K(dataset_path=config.opt.dataset_path,is_training=True,
                                        data_size=(config.opt.data_height, config.opt.data_width))
    else:
        dataset_object_train = None
        NotImplementedError('{} is not implemented'.format(config.opt.dataset_name))

    # Make dataloader with the above created preprocessing object
    train_loader = torch.utils.data.DataLoader(dataset_object_train,
                                               batch_size=config.opt.batch_size,
                                               shuffle=False, num_workers=0)
    return train_loader
