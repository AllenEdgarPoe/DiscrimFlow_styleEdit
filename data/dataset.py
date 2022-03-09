import os
import random
import torch
import torch.utils.data
from torchvision import transforms
from PIL import Image
from skimage import feature
import numpy as np

def create_fix_mask(tensor):
    mask = torch.zeros(tensor.size())
    num = mask.size(2) // 4
    mask[:, num:num * 3, num:num * 3] = 1
    return mask

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 5, W-5)
    bby1 = np.clip(cy - cut_h // 2, 5, H-5)
    bbx2 = np.clip(cx + cut_w // 2, 5, W-5)
    bby2 = np.clip(cy + cut_h // 2, 5, H-5)

    return bbx1, bby1, bbx2, bby2

def box2img(img):
    mask = torch.zeros(img.size())
    image = img.clone()
    bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), 0.3)
    mask[:, bbx1:bbx2, bby1:bby2] = 1
    image = image*(1-mask)
    return image, mask

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

class ImgDatasets(torch.utils.data.Dataset):
    def __init__(self, root_dir, files, mode='sketch', train=True):
        self.img_files = files_to_list(files)
        self.root_dir = root_dir
        self.train = train
        self.origin_transform_test = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Resize(64),
            # transforms.RandomHorizontalFlip()
        ])
        self.origin_transform_train = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip()
        ])
        self.gray_transform = transforms.Compose([
           transforms.Grayscale()
        ])
        self.ToTensor = transforms.ToTensor()
        random.seed(1234)
        random.shuffle(self.img_files)
        self.mode = mode

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.img_files[index])
        image = Image.open(img_name)
        if self.train:
            image = self.origin_transform_train(image)
        else:
            image = self.origin_transform_test(image)
        gray_img = self.gray_transform(image)
        image = self.ToTensor(image)
        gray_img = self.ToTensor(gray_img)
        if self.mode == 'gray':
            return (image, gray_img)
        elif self.mode == 'sketch':
            edges = feature.canny(gray_img.squeeze(0).numpy(), sigma=0.3)
            edges = torch.from_numpy(edges).type(torch.float)
            edges = edges.unsqueeze(0)
            return (image, edges)

        elif self.mode == "fix_inpainting":
            mask = create_fix_mask(image)
            cond_img = image * (1-mask)
            return (image, cond_img)

        elif self.mode == 'random_inpainting':
            cond_img, mask = box2img(image)
            return (image, cond_img)

if __name__ == "__main__":
    filename = '/home/jkshark/data/celeba'
    att_file = os.path.join(filename, 'list_attr_celeba.txt')

    lines = [line.rstrip() for line in open(att_file, 'r')][2:]
    train_data = [line.split()[0] for line in lines[2000:]]
    test_data = [line.split()[0] for line in lines[:2000]]
    with open("train_files.txt", 'w') as f:
        f.write(('\n').join(train_data))
    with open("test_files.txt", 'w') as f:
        f.write(('\n').join(test_data))


    # filename = "../train_files.txt"
    # att_file = "./list_attr_celeba.txt"
    # sample_size = 16
    # dataset = ImgDatasets("celeba_sample", filename, att_file)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=sample_size)
    # original, cond_img = next(iter(loader))
    # data = { "original": original,
    #          "cond_img": cond_img,
    # }
    # torch.save(data, "../inference_data/for_inference.pt")
