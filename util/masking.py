import torch

def create_fix_mask(tensor):
    mask = torch.zeros(tensor.size())
    num = mask.size(3) // 4
    mask[:, :, num:num * 3, num:num * 3] = 1
    return mask

def create_eye_mask(tensor):
    mask = torch.zeros((tensor.size()))
    mask[:, :, 20:40, 16:48] = 1
    return mask

def create_mouth_mask(tensor):
    mask = torch.zeros((tensor.size()))
    mask[:, :, 50:60, 18:46] = 1
    return mask

def create_random_mask(tensor):
    mask = torch.zeros(tensor.size())
    num = mask.size(2) // 4
    mask[:, num:num * 3, num:num * 3] = 1
    return mask