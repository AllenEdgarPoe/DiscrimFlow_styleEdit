import torch
import os
import torchvision
from models import Glow
from data.dataset import ImgDatasets
from torch.utils import data
from torchvision import transforms
from util.masking import create_eye_mask, create_mouth_mask, create_fix_mask
from util.make_attribute_vector import attribute_Dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt_path)
    num_channels, num_levels, num_steps = 128, 3, 8
    # num_channels, num_levels, num_steps = ckpt['parameter']
    net = Glow(num_channels=num_channels,
               num_levels=num_levels,
               num_steps=num_steps,
               mode='random_inpainting').to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, [0])
    net.load_state_dict(ckpt['net'])

    net.eval()

    testset = ImgDatasets(root_dir='/home/jkshark/data/celeba/images', files='./data/test_files.txt', mode=args.mode, train=False)
    testloader = data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=4)

    # attributeDataset = attribute_Dataset('/home/jkshark/data/celeba', ['Gray_Hair'])
    # attributeloader = data.DataLoader(attributeDataset, batch_size=1024, shuffle=False, num_workers=4)

    #style mixing test
    # test_iter = iter(testloader)
    # with torch.no_grad():
    #     for i in range(30):
    #         inpainting_style_mix(net, test_iter, device,args, i)

    # attribute test
    test_iter = iter(testloader)
    # attribute_iter = iter(attributeloader)
    with torch.no_grad():
        for i in range(30):
            inpainting_style_mix(net, test_iter, device, args, i)

def style_mix(net, testloader, device,args):
    test_iter = iter(testloader)
    img1, cond1 = next(test_iter)
    img1, cond1 = img1.to(device), cond1.to(device)
    img2, cond2 = next(test_iter)
    img2, cond2 = img2.to(device), cond2.to(device)
    img3, cond3 = next(test_iter)
    img3, cond3 = img3.to(device), cond3.to(device)

    #mix images
    mask = create_fix_mask(img1).to(device)
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

def inpainting_style_mix(net, test_iter, device,args, idx):
    img1, cond1 = next(test_iter)
    img1, cond1 = img1.to(device), cond1.to(device)
    img2, cond2 = next(test_iter)
    img2, cond2 = img2.to(device), cond2.to(device)
    img3, cond3 = next(test_iter)
    img3, cond3 = img3.to(device), cond3.to(device)

    blank_cond = torch.zeros(img1.size())

    #mix images
    z_1, _ =net(img1, blank_cond)   #color from img 1, but not only the color, basic features of the image also jams into the z
    z_2, _ = net(img2, blank_cond)
    z_3, _ =net(img3, blank_cond)
    random_z = torch.normal(mean=0, std=0.3, size=z_1.size())
    z_1 = z_1*0.7
    z_2 *=0.7

    reconstructed_r, _ = net(random_z, cond3, reverse=True)
    reconstructed_r = torch.sigmoid(reconstructed_r)

    reconstructed_1, _ = net(z_1, cond3, reverse=True)
    reconstructed_1 = torch.sigmoid(reconstructed_1)

    reconstructed_2, _ = net(z_2, cond3, reverse=True)
    reconstructed_2 = torch.sigmoid(reconstructed_2)

    original_list = []
    for i in range(img1.size(0)):
        original_list.append(img1[i].unsqueeze(0))
        original_list.append(img2[i].unsqueeze(0))
        original_list.append(img3[i].unsqueeze(0))
        original_list.append(cond3[i].unsqueeze(0))
        original_list.append(reconstructed_r[i].unsqueeze(0))
        original_list.append(reconstructed_1[i].unsqueeze(0))
        original_list.append(reconstructed_2[i].unsqueeze(0))
    original_list = torch.cat(original_list, 0)
    torchvision.utils.save_image(torchvision.utils.make_grid(original_list, nrow=7), os.path.join(args.output_dir, 'img_{}.png'.format(idx)))
    # torchvision.utils.save_image(torchvision.utils.make_grid(cond3, nrow=2), os.path.join(args.output_dir, 'inpainting_random_cond_{}.png').format(i))

def random_style_mix(net, test_iter, device,args, idx):
    img1, cond1= next(test_iter)
    img2, cond2 = next(test_iter)
    img3, cond3 = next(test_iter)

    eye_mask = create_eye_mask(img1)
    mouth_mask = create_mouth_mask(img1)
    blank_mask = torch.zeros((img1.size()))

    eye_cond1, eye_cond2, eye_cond3 = img1*(1-eye_mask), img2*(1-eye_mask), img3*(1-eye_mask)
    mouth_cond1, mouth_cond2, mouth_cond3 = img1*(1-mouth_mask), img2*(1-mouth_mask), img3*(1-mouth_mask)
    img1, eye_cond1, mouth_cond1 = img1.to(device), eye_cond1.to(device), mouth_cond1.to(device)
    img2, eye_cond2, mouth_cond2 = img2.to(device), eye_cond2.to(device), mouth_cond2.to(device)
    img3, eye_cond3, mouth_cond3 = img3.to(device), eye_cond3.to(device), mouth_cond3.to(device)

    eye_z_1, _ =net(img1, eye_cond1)
    eye_z_2, _ = net(img2, eye_cond2)
    eye_z_1 = eye_z_1 * 1.1
    eye_z_2 = eye_z_2 *1.1

    mouth_z_1, _ =net(img1, mouth_cond1)
    mouth_z_2, _ = net(img2, mouth_cond2)
    mouth_z_1 = mouth_z_1*1.1
    mouth_z_2 = mouth_z_2*1.1

    # z_1, _  = net(img1, blank_mask)
    # z_2, _ = net(img2, blank_mask)
    z_1 =torch.randn(img1.size(), dtype=torch.float32, device=device) * 0.7
    z_2 = torch.randn(img1.size(), dtype=torch.float32, device=device) * 0.7

    eye_reconstructed_1, _ = net(eye_z_1, eye_cond3, reverse=True)
    eye_reconstructed_1 = torch.sigmoid(eye_reconstructed_1)

    eye_reconstructed_2, _ = net(z_1, eye_cond3, reverse=True)
    eye_reconstructed_2 = torch.sigmoid(eye_reconstructed_2)

    mouth_reconstructed_1, _ = net(mouth_z_1, mouth_cond3, reverse=True)
    mouth_reconstructed_1 = torch.sigmoid(mouth_reconstructed_1)

    mouth_reconstructed_2, _ = net(z_2, mouth_cond3, reverse=True)
    mouth_reconstructed_2 = torch.sigmoid(mouth_reconstructed_2)

    original_list = []
    for i in range(img1.size(0)):
        original_list.append(img1[i].unsqueeze(0))
        original_list.append(img2[i].unsqueeze(0))
        original_list.append(img3[i].unsqueeze(0))
        original_list.append(eye_cond3[i].unsqueeze(0))
        original_list.append(eye_reconstructed_1[i].unsqueeze(0))
        original_list.append(eye_reconstructed_2[i].unsqueeze(0))
        original_list.append(mouth_cond3[i].unsqueeze(0))
        original_list.append(mouth_reconstructed_1[i].unsqueeze(0))
        original_list.append(mouth_reconstructed_2[i].unsqueeze(0))

    original_list = torch.cat(original_list, 0)
    torchvision.utils.save_image(torchvision.utils.make_grid(original_list, nrow=9), os.path.join(args.output_dir, 'img_{}.png'.format(idx)))
    # torchvision.utils.save_image(torchvision.utils.make_grid(cond3, nrow=2), os.path.join(args.output_dir, 'inpainting_random_cond_{}.png').format(i))

def attribute_manipulation(net, test_iter, attr_iter, device, args, idx):
    img, non_img = next(attr_iter)
    mask = create_fix_mask(img)
    cond, non_cond = img*(1-mask), non_img*(1-mask)

    img, non_img = img.to(device), non_img.to(device)
    cond, non_cond = cond.to(device), non_cond.to(device)

    z_pos, _ = net(img, cond)
    z_neg, _ = net(non_img, non_cond)

    z_pos1= torch.mean(z_pos, 0).unsqueeze(0)
    z_neg1 = torch.mean(z_neg, 0).unsqueeze(0)

    z_pos = z_pos1-z_neg1
    z_neg = z_neg1-z_pos1

    img2, cond2 = next(test_iter)
    img2, cond2 = img2.to(device), cond2.to(device)

    rec1, _ = net(z_pos.repeat(img2.size(0), 1, 1, 1)*2.0, cond2, reverse=True)
    rec1 = torch.sigmoid(rec1)
    rec2, _ = net(z_neg.repeat(img2.size(0), 1, 1, 1)*2.0, cond2, reverse=True)
    rec2 = torch.sigmoid(rec2)

    original_list = []
    for i in range(5):
        original_list.append(img2[i].unsqueeze(0))
        original_list.append(cond2[i].unsqueeze(0))
        original_list.append(rec1[i].unsqueeze(0))
        original_list.append(rec2[i].unsqueeze(0))

    original_list = torch.cat(original_list, 0)
    torchvision.utils.save_image(torchvision.utils.make_grid(original_list, nrow=4), os.path.join(args.output_dir, 'img_{}.png'.format(idx)))

def attribute_manipulation2(net, test_iter, attr_iter, device, args, idx):
    img, non_img = next(attr_iter)
    mask = create_fix_mask(img)
    cond, non_cond = img*(1-mask), non_img*(1-mask)

    img, non_img = img.to(device), non_img.to(device)
    cond, non_cond = cond.to(device), non_cond.to(device)

    z_pos, _ = net(img, cond)
    z_neg, _ = net(non_img, non_cond)

    z_pos= torch.mean(z_pos, 0).unsqueeze(0)
    z_neg = torch.mean(z_neg, 0).unsqueeze(0)

    img1, cond1 = next(test_iter)
    img1, cond1 = img1.to(device), cond1.to(device)
    img2, cond2 = next(test_iter)
    img2, cond2 = img2.to(device), cond2.to(device)
    img3, cond3 = next(test_iter)
    img3, cond3 = img3.to(device), cond3.to(device)

    z_1, _ = net(img1, cond1)
    z_2, _ = net(img2, cond2)

    # z_pos = torch.nn.Softmax()(z_pos)
    # z_pos = torchvision.transforms.Normalize(0,1)(z_pos)
    z_pos2 = z_pos - z_neg
    z_neg2 = z_neg - z_pos
    z_pos_1 = z_1+ z_pos2.repeat(img1.size(0), 1, 1, 1)*3
    z_pos_2 = z_2+ z_pos2.repeat(img2.size(0), 1, 1, 1)*3


    # z_pos_2 = torchvision.transforms.Normalize(0,1)(z_pos_2)
    z_neg_1 = z_1
    z_neg_2 = z_2


    rec_pos_1, _ = net(z_pos_1*0.7, cond3, reverse=True)
    rec_pos_1 = torch.sigmoid(rec_pos_1)

    rec_pos_2, _ = net(z_pos_2*0.7, cond3, reverse=True)
    rec_pos_2 = torch.sigmoid(rec_pos_2)

    rec_neg_1, _ = net(z_neg_1*0.7, cond3, reverse=True)
    rec_neg_1 = torch.sigmoid(rec_neg_1)

    rec_neg_2, _ = net(z_neg_2*0.7, cond3, reverse=True)
    rec_neg_2 = torch.sigmoid(rec_neg_2)

    original_list = []
    for i in range(5):
        original_list.append(img1[i].unsqueeze(0))
        original_list.append(img2[i].unsqueeze(0))
        original_list.append(img3[i].unsqueeze(0))
        original_list.append(cond3[i].unsqueeze(0))
        original_list.append(rec_pos_1[i].unsqueeze(0))
        original_list.append(rec_neg_1[i].unsqueeze(0))
        original_list.append(rec_pos_2[i].unsqueeze(0))
        original_list.append(rec_neg_2[i].unsqueeze(0))


    original_list = torch.cat(original_list, 0)
    torchvision.utils.save_image(torchvision.utils.make_grid(original_list, nrow=8), os.path.join(args.output_dir, 'img_{}.png'.format(idx)))

def attribute_manipulation3(net, test_iter, attr_iter, device, args, idx):
    img, non_img = next(attr_iter)
    mask = create_fix_mask(img)
    cond, non_cond = img*(1-mask), non_img*(1-mask)

    img, non_img = img.to(device), non_img.to(device)
    cond, non_cond = cond.to(device), non_cond.to(device)

    z_pos, _ = net(img, cond)
    z_neg, _ = net(non_img, non_cond)

    pos_mean, pos_var = torch.mean(z_pos), torch.var(z_pos)
    neg_mean, neg_var = torch.mean(z_neg), torch.var(z_neg)

    img2, cond2 = next(test_iter)
    img2, cond2 = img2.to(device), cond2.to(device)

    z_pos = torch.normal(mean=float(pos_mean), std=float(pos_var), size=img2.size())
    z_neg = torch.normal(mean=float(neg_mean), std=float(neg_var), size=img2.size())

    rec1, _ = net(z_pos*0.5, cond2, reverse=True)
    rec1 = torch.sigmoid(rec1)
    rec2, _ = net(z_neg*0.5, cond2, reverse=True)
    rec2 = torch.sigmoid(rec2)

    original_list = []
    for i in range(5):
        original_list.append(img2[i].unsqueeze(0))
        original_list.append(cond2[i].unsqueeze(0))
        original_list.append(rec1[i].unsqueeze(0))
        original_list.append(rec2[i].unsqueeze(0))

    original_list = torch.cat(original_list, 0)
    torchvision.utils.save_image(torchvision.utils.make_grid(original_list, nrow=4), os.path.join(args.output_dir, 'img_{}.png'.format(idx)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='fix_inpainting')
    parser.add_argument('--ckpt_path', default='ckpts/fix_box/best.pth.tar')
    parser.add_argument('--output_dir', default="inference_data/tmp")

    args = parser.parse_args()
    main(args)