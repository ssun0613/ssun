import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy import io
import glob


class WM_811K():
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
    from utils.common import tensor2numpy
    import os
    import dataset
    from options.config import Config
    import numpy as np

    config = Config()
    train_loader = dataset.create_dataloader(config)
#
#     save_pth_debug_img = '/Users/ssun/Documents/debug_img/'
#     if not os.path.exists(save_pth_debug_img):
#         os.makedirs(save_pth_debug_img, exist_ok=True)
#
    for batch_id, data in enumerate(train_loader, 1):
        for i in range(len(data['image'])):
            data_img = tensor2numpy(data['image'][i])
            # Image.fromarray(data_img).save(save_pth_debug_img + 'input.png')
            print(data['label'][i])
        print("-----------" +str(batch_id)+ "-----------")
