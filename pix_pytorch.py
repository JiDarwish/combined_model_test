import os
import torch
from PIL import Image
from skimage.transform import resize
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pix import rgb_to_lab


class ImageDataset(Dataset):
    def __init__(self, paths, im_size=256, crop_size=224, split='train'):
        self.split = split
        self.paths = paths
        self.total_imgs = os.listdir(paths)

        # for the train set, data augmentation with RandomHorizontalFlip
        if self.split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((im_size, im_size)),
                transforms.RandomHorizontalFlip(),
            ])
        # for the test set, make sure that the image are at the right size only
        elif self.split == 'test':
            self.transforms = transforms.Resize((im_size, im_size))

        # this preprocessing is for the input of the classifier for the gamma model
        # it is not used for the other models
        self.preprocess = transforms.Compose([
            transforms.Resize(im_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, index):
        # open the image in RGB
        img_loc = os.path.join(self.paths, self.total_imgs[index])
        img = Image.open(img_loc).convert('RGB')

        # preprocess the images for the classifier (gamma model)
        input_tensor = self.preprocess(img)

        # augment the images and transform them to LAB
        img_input = self.transforms(img)
        img_lab = rgb_to_lab(img_input)

        # return L, ab and embed separately
        return {'L': transforms.ToTensor()(img_lab['L']), 'ab': transforms.ToTensor()(img_lab['ab'])}, input_tensor

def make_dataloaders(batch_size=16, im_size=256, split='Train', n_workers=2, pin_memory=True, shuffle=True, **kwargs):
    dataset = ImageDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader