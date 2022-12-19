# How to add the parent directories
import sys
sys.path.append('..')

import torch
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import glob

class single_type():
    def __init__(self, dataset_path, data_size):
        self.image_size = data_size
        self.dataset_dir = dataset_path
        self._toTensor_image = self._toTensor_image()
        self._toTensor_label = self._toTensor_label()

        self.data_image = self.dataset_image()
        self.data_label = self.dataset_label()
        self.dataset_size = len(self.data_label)

    def dataset_image(self):
        data_image = sorted(glob.glob(self.dataset_dir + 'image/*.npz'))
        return data_image

    def dataset_label(self):
        label_re = sorted(glob.glob(self.dataset_dir + 'label/*.npz'))
        return label_re

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # image_split = np.load(self.data_image[index % self.dataset_size])
        image_split = np.load(self.data_image[index])
        image_split = image_split['arr_0']
        # label_split = np.load(self.data_label[index % self.dataset_size])
        label_split = np.load(self.data_label[index])
        label_split = label_split['arr_0']
        image_split = np.expand_dims(image_split, axis=0)
        label_split = np.expand_dims(label_split, axis=0)

        data_zero_image = np.zeros([image_split.shape[1], image_split.shape[2], 3])
        data_zero_image[np.where(image_split.squeeze(axis=0) == 0)] = (0, 0, 0)
        data_zero_image[np.where(image_split.squeeze(axis=0) == 1)] = (71, 100, 100)
        data_zero_image[np.where(image_split.squeeze(axis=0) == 2)] = (255, 228, 0)
        image_split = data_zero_image

        image = self._toTensor_image(Image.fromarray(image_split.astype(dtype=np.uint8)).convert("RGB"))
        # image = self._toTensor_image(Image.fromarray(image_split.astype(dtype=np.uint8)))

        label_split = torch.tensor(label_split)
        label = label_split.squeeze(dim=0)

        return {'image': image, 'label': label}

    def _toTensor_image(self):
        toTensor_image = transforms.Compose([transforms.Resize(self.image_size),
                                       transforms.ToTensor()])

        # toTensor_image = transforms.Compose([transforms.Resize(self.image_size),
        #                                      transforms.RandomVerticalFlip(),
        #                                      transforms.RandomHorizontalFlip(),
        #                                      transforms.RandomRotation(degrees=(0, 90)),
        #                                      transforms.ToTensor()])
        return toTensor_image

    def _toTensor_label(self):
        toTensor_label = transforms.ToTensor()
        return toTensor_label

if __name__ == '__main__': # class 내부에서 바로 디버깅을 할수있음
    from utils.common import tensor2numpy
    import torchvision.transforms as transforms
    import numpy as np
    import os
    from options.config import Config

    config = Config()
    dataset_object_train = single_type(dataset_path=config.opt.dataset_path,
                                       data_size=(config.opt.data_height, config.opt.data_width),)

    save_pth_debug_img = '/home/mskim/'
    if not os.path.exists(save_pth_debug_img):
        os.makedirs(save_pth_debug_img, exist_ok=True)

    # out = dataset_object_train.__getitem__(31259)
    #
    # for i in range(0, 45000 ,1):
    out = dataset_object_train.__getitem__(28651)
    out_image = tensor2numpy(out['image'])
    # out_image = out_image.squeeze(axis=2)
    out_label = out['label']
    print(out_label)

    # sns.heatmap(map_1, annot=True, fmt='.2f', cmap='Blues', ax=axs[0])
    # np.savez('/storage/mskim/WDM/single_data_1/test_aug/image/' + str('R') + str(i-883), out_image)
    # np.savez('/storage/mskim/WDM/single_data_1/test_aug/label/' + str('R') + str(i-883), out_label)








