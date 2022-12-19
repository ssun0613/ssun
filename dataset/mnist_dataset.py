import sys, os
sys.path.append(os.pardir)

import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob

class mnist():
    def __init__(self, dataset_path, data_size):
        self.image_size = data_size
        self.dataset_dir = dataset_path
        self._toTensor_image = self._toTensor_image()
        self._toTensor_label = self._toTensor_label()

        self.data_image = self.dataset_image()
        self.dataset_size = len(self.data_image)
        self.n_class= 10

    def dataset_image(self):
        data_image = sorted(glob.glob(self.dataset_dir+'**/*.png', recursive=True))
        # glob를 이용하여 하위 디렉토리를 검색하기 위해서는 recursive=True로 해주고, '**'를 이용하면 하위 디렉토리에서 파일리스트를 가지고 올수 있음
        return data_image

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        image = self.data_image[index % self.dataset_size]
        label = image.split('/')[6]
        # label = image.split('/')[9]
        image = self._toTensor_image(Image.open(image).convert('L'))
        label = self.one_hot_encoding(int(label), self.n_class)
        label = self._toTensor_label(label)
        label = label.squeeze(dim=0).squeeze(dim=0)

        return {'image': image, 'label': label}

    def _toTensor_image(self):
        toTensor_image = transforms.Compose([transforms.Resize(self.image_size),
                                             transforms.RandomVerticalFlip(),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomRotation(degrees=(0, 90)),
                                             transforms.ToTensor()])
        return toTensor_image

    def _toTensor_label(self):
        toTensor_label = transforms.ToTensor()
        return toTensor_label

    def one_hot_encoding(self, t, n_class):
        x_label= np.zeros([1, n_class])
        x_label[0, t]= 1
        return x_label

if __name__ == '__main__':  # class 내부에서 바로 디버깅을 할수있음
    from utils.common import tensor2numpy
    import torchvision.transforms as transforms
    import numpy as np
    import os
    from options.config import Config

    config = Config()
    dataset_object_train = mnist(dataset_path=config.opt.dataset_path,
                                data_size=(config.opt.data_height, config.opt.data_width),)

    out = dataset_object_train.__getitem__(65)