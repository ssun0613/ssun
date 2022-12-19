import sys
sys.path.append("..")
import numpy as np
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob

class horse_human():
    def __init__(self, dataset_path, data_size, is_training=True):
        self.image_size = data_size
        self.is_training = is_training
        self._toTensor = self._toTensor()
        self.transform = self._make_transformer(is_training=self.is_training)

        self.dataset_dir = dataset_path
        self.img_paths = sorted(glob.glob(self.dataset_dir+'**/*.png', recursive=True))
        self.dataset_size = len(self.img_paths)
        self.n_class = 2
        self.class_index = {'horses': 0, 'humans': 1}

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        img_path = self.img_paths[index % self.dataset_size]
        image = self._toTensor(Image.open(img_path).convert('RGB'))

        label_path = img_path.split('/')[6]
        if label_path =='horses':
            label_path = np.squeeze(self.one_hot_encoding('horses', self.n_class), axis=0)
        elif label_path =='humans':
            label_path = np.squeeze(self.one_hot_encoding('humans', self.n_class), axis=0)
        else:
            print("check label_path")
        label = label_path

        return {'image': image, 'label': label}

    def _toTensor(self):
        toTensor = transforms.Compose([transforms.ToTensor()])
        return toTensor

    def _make_transformer(self, is_training=True):
        if is_training:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-0, 0))])
        else:
            transform = None
        return transform

    def one_hot_encoding(self, t, n_class):
        x_label= np.zeros([1, n_class])
        x_label[0, self.class_index[t]] = 1
        return x_label

if __name__ == '__main__':
    from utils.common import tensor2numpy
    import os
    import sys
    sys.path.append("..")
    import dataset
    from options.config import Config

    config = Config()
    train_loader = horse_human(dataset_path=config.opt.dataset_path_train,
                               data_size=(config.opt.data_height, config.opt.data_width),
                               is_training=True)

    cnt_human = 500
    cnt_horse = 527
    for batch_id, data in enumerate(train_loader, 1):
        for i in range(len(data['img'])):

            data_img = tensor2numpy(data['img'][i])
            data_label = tensor2numpy(data['label'][i])
            # if data_label == 1:
            #     if cnt_human <= 500*0.8:
            #         # dataset 경로
            #     else
            #         # testdatset 경로로
            #    cnt_human += 1
            # elif data_label == 0:
            #     cnt_horse += 1
            Image.fromarray(data_img).save(save_pth_debug_img + 'input.png')
            Image.fromarray(data_label).save(save_pth_debug_img + 'label.png')

    print(cnt_human, cnt_horse)

