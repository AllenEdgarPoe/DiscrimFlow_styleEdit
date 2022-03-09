import torch.utils.data as data
from torchvision import transforms
import os, random
from PIL import Image

class attribute_Dataset(data.Dataset):
    def __init__(self, root_dir, selected_attrs):
        self.image_dir = os.path.join(root_dir, 'images')
        self.selected_attrs = selected_attrs
        self.attr_path = os.path.join(root_dir, 'list_attr_celeba.txt')
        self.attr2idx = {}
        self.idx2attr = {}
        self.image_list = []
        self.non_image_list = []

        self.preprocess()


        if self.non_image_list>self.image_list:
            self.non_image_list = self.non_image_list[:len(self.image_list)]
        else:
            self.image_list = self.image_list[:len(self.non_image_list)]

        self.num_images = len(self.image_list)
        self.transform = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Resize(64),
            transforms.ToTensor()
            ])

    def preprocess(self):
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

            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                if values[idx] == '1':
                    self.image_list.append(filename)
                else:
                    self.non_image_list.append(filename)

    def __getitem__(self, index):
        filename = self.image_list[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        image = self.transform(image)

        non_filename = self.non_image_list[index]
        non_image = Image.open(os.path.join(self.image_dir, non_filename))
        non_image = self.transform(non_image)

        return image, non_image

    def __len__(self):
        return len(self.image_list)