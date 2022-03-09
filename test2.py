import torch
import os
import torchvision
from models.glow.glow2 import Glow
from data.dataset2 import ImgDatasets
from torch.utils import data
from torchvision import transforms

def create_labels(c_org, c_dim=5, selected_attrs=None, gender=True):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
            hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        c_trg = c_org.clone()
        if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
            c_trg[:, i] = 1
            for j in hair_color_indices:
                if j != i:
                    c_trg[:, j] = 0
        elif i == 0 and gender:
            c_trg[:, i] = 1
        elif i==0 and gender==False:
            c_trg[:,i] = 0
        else:
            c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

        c_trg_list.append(c_trg)
    return c_trg_list


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)
    net = Glow(cond1=3, cond2=1,
               num_channels=128,
               num_levels=4,
               num_steps=16,
               mode='fix_inpainting').to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, [0,1])

    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in ckpt['net'].items():
    #     name = k[7:]
    #     new_state_dict[name] = v
    net.load_state_dict(ckpt['net'])

    net.eval()

    root_file = args.data_dir
    att_file = os.path.join(root_file, 'list_attr_celeba.txt')
    img_file = os.path.join(root_file, 'images')
    selected_attrs = ['Male']
    testset = ImgDatasets(img_file, att_file, selected_attrs, mode='fix_inpainting', train=False)
    testloader = data.DataLoader(testset, batch_size=5, shuffle=False)
    #style mixing test
    test_iter = iter(testloader)
    with torch.no_grad():
        for i in range(20):
            inpainting_style_mix2(net, test_iter, device,args, i)

def style_mix(net, testloader, device,args):
    test_iter = iter(testloader)
    img1, cond1 = next(test_iter)
    img1, cond1 = img1.to(device), cond1.to(device)
    img2, cond2 = next(test_iter)
    img2, cond2 = img2.to(device), cond2.to(device)
    img3, cond3 = next(test_iter)
    img3, cond3 = img3.to(device), cond3.to(device)

    #mix images
    mask = create_mask(img1).to(device)
    new_img = img1*mask + img2*(1-mask)
    # new_img = img2*0.8+img1*0.2
    new_img_cond = transforms.Grayscale()(new_img)
    z, _ = net(new_img, new_img_cond)
    z_1, _ =net(img1, cond1)   #color from img 1, but not only the color, basic features of the image also jams into the z
    z_2, _ = net(img2, cond2)
    z_3, _ =net(img3, cond3)
    # new_z = z_2*mask + z_1*(1-mask)
    reconstructed, _ = net(z_3, new_img_cond, reverse=True)
    reconstructed = torch.sigmoid(reconstructed)

    # reconstructed_t, _ = net(z_2, new_img_cond, reverse=True)
    # reconstructed_t = torch.sigmoid(reconstructed_t)

    original_list = []
    for i in range(img1.size(0)):
        original_list.append(img1[i].unsqueeze(0))
        original_list.append(img2[i].unsqueeze(0))
        original_list.append(new_img[i].unsqueeze(0))
        original_list.append(img3[i].unsqueeze(0))
        original_list.append(reconstructed[i].unsqueeze(0))
        # original_list.append(reconstructed_t[i].unsqueeze(0))
    original_list = torch.cat(original_list, 0)
    torchvision.utils.save_image(torchvision.utils.make_grid(original_list, nrow=5), os.path.join(args.output_dir, 'inpainting_ori_img.png'))
    torchvision.utils.save_image(torchvision.utils.make_grid(new_img_cond, nrow=2), os.path.join(args.output_dir, 'inpainting_cond_img.png'))

def create_mask(tensor):
    mask = torch.zeros(tensor.size())
    num = mask.size(2) // 4
    mask[:, num:num * 3, num:num * 3] = 1
    return mask

def sample(net, test_iter, device, args, idx):
    img, cond_m, cond_att = next(test_iter)
    img, cond_m, cond_att = img.to(device), cond_m.to(device), cond_att.to(device)

    B, C, W, H = img.shape
    z1 = torch.randn((B, 3, 64, 64), dtype=torch.float32, device=device) * 0.6
    z2 = torch.randn((B, 3, 64, 64), dtype=torch.float32, device=device) * 0.6

    # set target domain labels
    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    male_target_att = create_labels(cond_att, 5, selected_attrs, gender='Male')
    female_target_att = create_labels(cond_att, 5, selected_attrs, gender='Female')
    male_target_att = male_target_att[3].to(device)
    female_target_att = female_target_att[3].to(device)

    tmp = []

    rec1, _ = net(z1, cond_m, male_target_att, reverse=True)
    rec1 = torch.sigmoid(rec1)
    tmp.append(rec1)

    rec2, _ = net(z1, cond_m, female_target_att, reverse=True)
    rec2 = torch.sigmoid(rec2)
    tmp.append(rec2)

    original_list = []
    for i in range(img.size(0)):
        original_list.append(img[i].unsqueeze(0))
        original_list.append(cond_m[i].unsqueeze(0))
        original_list.append(rec1[i].unsqueeze(0))
        original_list.append(rec2[i].unsqueeze(0))
    original_list = torch.cat(original_list, 0)
    torchvision.utils.save_image(torchvision.utils.make_grid(original_list, nrow=4), os.path.join(args.output_dir, 'img_{}.png'.format(idx)))


def inpainting_style_mix(net, test_iter, device,args, idx):
    img1, cond1_m, cond1_att = next(test_iter)
    img1, cond1_m, cond1_att = img1.to(device), cond1_m.to(device), cond1_att.to(device)
    img2, cond2_m, cond2_att = next(test_iter)
    img2, cond2_m, cond2_att = img2.to(device), cond2_m.to(device), cond2_att.to(device)
    img3, cond3_m, cond3_att = next(test_iter)
    img3, cond3_m, cond3_att = img3.to(device), cond3_m.to(device), cond3_att.to(device)

    # set target domain labels
    selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    male_target_att = create_labels(cond3_att, 5, selected_attrs, gender='Male')
    female_target_att = create_labels(cond3_att, 5, selected_attrs, gender='Female')
    male_target_att = male_target_att[3].to(device)
    female_target_att = female_target_att[3].to(device)

    #mix images
    z1, _ =net(img1, cond1_m, cond1_att)   #color from img 1, but not only the color, basic features of the image also jams into the z

    z2, _ = net(img2, cond2_m, cond2_att)

    z3, _ =net(img3, cond3_m, cond3_att)

    reconstructed_1, _ = net(z1, cond3_m, male_target_att,  reverse=True)
    reconstructed_1_1 = torch.sigmoid(reconstructed_1)
    reconstructed_1, _ = net(z1, cond3_m, female_target_att,  reverse=True)
    reconstructed_1_2 = torch.sigmoid(reconstructed_1)

    reconstructed_2, _ = net(z2, cond3_m, male_target_att,  reverse=True)
    reconstructed_2_1 = torch.sigmoid(reconstructed_2)
    reconstructed_2, _ = net(z2, cond3_m, female_target_att,  reverse=True)
    reconstructed_2_2 = torch.sigmoid(reconstructed_2)

    original_list = []
    for i in range(img1.size(0)):
        original_list.append(img1[i].unsqueeze(0))
        original_list.append(img2[i].unsqueeze(0))
        original_list.append(img3[i].unsqueeze(0))
        original_list.append(cond3_m[i].unsqueeze(0))
        original_list.append(reconstructed_1_1[i].unsqueeze(0))
        original_list.append(reconstructed_1_2[i].unsqueeze(0))
        original_list.append(reconstructed_2_1[i].unsqueeze(0))
        original_list.append(reconstructed_2_2[i].unsqueeze(0))
    original_list = torch.cat(original_list, 0)
    torchvision.utils.save_image(torchvision.utils.make_grid(original_list, nrow=8), os.path.join(args.output_dir, 'img_{}.png'.format(idx)))
    # torchvision.utils.save_image(torchvision.utils.make_grid(cond3, nrow=2), os.path.join(args.output_dir, 'inpainting_random_cond_{}.png').format(i))

def inpainting_style_mix2(net, test_iter, device,args, idx):
    img1, cond1_m, cond1_att = next(test_iter)
    img1, cond1_m, cond1_att = img1.to(device), cond1_m.to(device), cond1_att.to(device)
    img2, cond2_m, cond2_att = next(test_iter)
    img2, cond2_m, cond2_att = img2.to(device), cond2_m.to(device), cond2_att.to(device)
    img3, cond3_m, cond3_att = next(test_iter)
    img3, cond3_m, cond3_att = img3.to(device), cond3_m.to(device), cond3_att.to(device)

    # set target domain labels
    selected_attrs = ['Male']
    male_target_att = create_labels(cond3_att, 1, selected_attrs, gender=True)
    female_target_att = create_labels(cond3_att, 1, selected_attrs, gender=False)
    male_target_att = male_target_att[0].to(device)
    female_target_att = female_target_att[0].to(device)

    #mix images
    cond1_m = torch.zeros(cond1_m.size()).to(device)
    z1, _ =net(img1, cond1_m, cond1_att)   #color from img 1, but not only the color, basic features of the image also jams into the z

    z2, _ = net(img2, cond2_m, cond2_att)

    z3, _ =net(img3, cond3_m, cond3_att)

    reconstructed_1, _ = net(z1*0.7, cond3_m, male_target_att,  reverse=True)
    reconstructed_1_1 = torch.sigmoid(reconstructed_1)
    reconstructed_1, _ = net(z1*0.7, cond3_m, female_target_att,  reverse=True)
    reconstructed_1_2 = torch.sigmoid(reconstructed_1)

    reconstructed_2, _ = net(z2*0.7, cond3_m, male_target_att,  reverse=True)
    reconstructed_2_1 = torch.sigmoid(reconstructed_2)
    reconstructed_2, _ = net(z2*0.7, cond3_m, female_target_att,  reverse=True)
    reconstructed_2_2 = torch.sigmoid(reconstructed_2)

    original_list = []
    for i in range(img1.size(0)):
        original_list.append(img1[i].unsqueeze(0))
        original_list.append(img2[i].unsqueeze(0))
        original_list.append(img3[i].unsqueeze(0))
        original_list.append(cond3_m[i].unsqueeze(0))
        original_list.append(reconstructed_1_1[i].unsqueeze(0))
        original_list.append(reconstructed_1_2[i].unsqueeze(0))
        original_list.append(reconstructed_2_1[i].unsqueeze(0))
        original_list.append(reconstructed_2_2[i].unsqueeze(0))
    original_list = torch.cat(original_list, 0)
    torchvision.utils.save_image(torchvision.utils.make_grid(original_list, nrow=8), os.path.join(args.output_dir, 'img_{}.png'.format(idx)))
    # torchvision.utils.save_image(torchvision.utils.make_grid(cond3, nrow=2), os.path.join(args.output_dir, 'inpainting_random_cond_{}.png').format(i))
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/jkshark/data/celeba', type=str)
    parser.add_argument('--ckpt_path', default='ckpts/gender_cond/best.pth.tar')
    parser.add_argument('--output_dir', default="inference_data/two_cond/gender")

    args = parser.parse_args()
    main(args)