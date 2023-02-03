import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy import io
import glob


class single_mat():
    def __init__(self, dataset_path, data_size, is_training=True):
        self.image_size = data_size
        self.dataset_dir = dataset_path
        self.is_training = is_training
        self._toTensor = self._toTensor()
        self.data_load = self.data_load()
        self.dataset_size = len(self.data_load)

    def data_load(self):
        data_load = sorted(glob.glob(self.dataset_dir + '*.mat'))
        return data_load


    def __len__(self):
        return self.dataset_size

    # train_multi_98, train_multi_96, train_multi_90, train_multi_88, train_multi_82, train_multi_80, train_multi_74, train_multi_72, train_multi_66, train_multi_42,
    # train_multi_40, train_multi_34, train_multi_26, train_multi_24, train_multi_18, train_multi_170, train_multi_168, train_multi_162, train_multi_160, train_multi_154
    # train_multi_144, train_multi_138, train_multi_136, train_multi_130, train_multi_10, train_multi_106, train_multi_104, train_multi_152, train_multi_146
    # train_center, train_donut, train_loc, train_edgering, train_edgeloc, train_nearfull, train_random, train_scratch, train_none
    def __getitem__(self, index):
        data_one = io.loadmat(self.data_load[index % self.dataset_size])
        image_one = data_one['data_wdm']['waferDefectMap'][0][0]
        label_one = data_one['data_wdm']['label_1'][0][0].transpose()
        image_one = np.expand_dims(image_one, axis=0)

        data_zero = np.zeros([image_one.shape[1], image_one.shape[2], 3])
        data_zero[np.where(image_one.squeeze(axis=0) == 0)] = (0, 0, 0)
        data_zero[np.where(image_one.squeeze(axis=0) == 1)] = (71, 100, 100)
        data_zero[np.where(image_one.squeeze(axis=0) == 2)] = (255, 228, 0)
        image_one = data_zero # image를 RGB형태로 변환하기 위한 code --> training에는 필요없음, wandb에서 image출력하기위해서는 필요

        label = torch.tensor(label_one)
        label = label.squeeze(dim=0)

        image = self._toTensor(Image.fromarray(image_one.astype(dtype=np.uint8)).convert("RGB"))

        return {'image': image, 'label': label}

    def _toTensor(self):
        toTensor = transforms.Compose([transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
                                       transforms.ToTensor()])
        return toTensor

if __name__ == '__main__': # class 내부에서 바로 디버깅을 할수있음
    import os
    import sys
    sys.path.append('..')
    import dataset
    from options.config import Config
    import numpy as np

    config = Config()
    dataset_object_train = single_mat(dataset_path=config.opt.dataset_path_train, is_training=True,
                                        data_size=(config.opt.data_height, config.opt.data_width))


    out = dataset_object_train.__getitem__(28651)

