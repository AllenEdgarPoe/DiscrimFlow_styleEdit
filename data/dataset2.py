import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random

def create_mask(tensor):
    mask = torch.zeros(tensor.size())
    num = mask.size(2) // 16
    mask[:, num*3:num * 13, num*3:num * 13] = 1
    # Eye
    # mask[:, 25:40,16:48] = 1
    #Mouth
    # mask[:, 50:60, 18:46] = 1
    return mask

class ImgDatasets(data.Dataset):
    def __init__(self, image_dir, attr_path, selected_attrs, mode, train=True):
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.mode = mode
        self.train = train
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if self.train:
            self.num_images = len(self.train_dataset)
            self.transform = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        else:
            self.num_images = len(self.test_dataset)
            self.transform = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Resize(64),
            transforms.ToTensor()
        ])

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        dataset = self.train_dataset if self.train else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        image = self.transform(image)
        if self.mode == "fix_inpainting":
            mask = create_mask(image)
            cond1 = image*(1-mask)
            cond2 = torch.FloatTensor(label)
            cond2 = cond2.view(cond2.size(0), 1, 1)
            cond2 = cond2.repeat(1, image.size(1), image.size(2))
        else:
            raise Exception
        return image, cond1, cond2

    def __len__(self):
        return self.num_images


if __name__ =="__main__":
    root_file = '/home/jkshark/data/celeba'
    att_file = os.path.join(root_file, 'list_attr_celeba.txt')
    img_file = os.path.join(root_file, 'images')
    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']

    dataset = ImgDatasets(img_file, att_file, selected_attrs, mode='fix_inpainting', train=False)
    dataloader = iter(data.DataLoader(dataset, batch_size=4))
    img, cond1, cond2 = next(dataloader)
    from torchvision.utils import make_grid, save_image
    print(cond2)
    save_image(make_grid(torch.cat((img, cond1),0), nrow=2), './test.jpg')