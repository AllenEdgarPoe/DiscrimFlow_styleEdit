import torch
import numpy as np
from data.dataset import ImgDatasets
from torch.utils import data

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
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
    for i in range(img.size(0)):
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), 0.3)
        img[i, :, bbx1:bbx2, bby1:bby2] = 0
    return img

def test_mask():
    testset = ImgDatasets(root_dir='/home/jkshark/data/celeba/images', files='./data/test_files.txt', mode='inpainting',
                          train=False)
    testloader = data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)
    img, cond = next(iter(testloader))
    result = box2img(img)
    from torchvision.utils import make_grid, save_image
    save_image(make_grid(result, nrow=4), '../test.png')

if __name__ == "__main__":
    test_mask()