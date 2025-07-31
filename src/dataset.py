from torch.utils.data import Dataset
from torchvision import transforms
import os
import cv2
import random


class DiffusionDataset(Dataset):
    def __init__(self, data_dir, image_size=1024):
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_list = [ file for file in os.listdir(os.path.join(data_dir, 'images')) if file.endswith('.jpg') or file.endswith('.png')]
        self.image_list = sorted(self.image_list, key=lambda x: int(x.split('.')[0]))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        try:
            img = cv2.imread(os.path.join(self.data_dir, 'images', self.image_list[idx]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            w, h = self.image_size, self.image_size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32

            # to tensor of [0,1]
            img = transforms.ToTensor()(img)

            # random resize crop
            img = transforms.RandomResizedCrop((new_h, new_w), scale=(0.5, 1.0))(img)
            visualizable_image = img
            
            # to diffusion tensor
            img = img * 2 - 1

            # get the prompt
            with open(os.path.join(self.data_dir, 'prompts', self.image_list[idx].split('.')[0]+'.txt'), 'r') as f:
                prompt = f.readline().strip()


            return {'image': img, 'visualizable_image': visualizable_image, 'prompt': prompt}
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))