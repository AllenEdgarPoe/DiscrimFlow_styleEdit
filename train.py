"""Train Glow on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
from data.dataset import ImgDatasets
import util

from models import Glow
from models.discriminator import Discriminator
from tqdm import tqdm


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # Set up main device and scale batch size
    device = 'cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    root_file = args.data_dir
    att_file = os.path.join(root_file, 'list_attr_celeba.txt')
    img_file = os.path.join(root_file, 'images')

    trainset = ImgDatasets(root_dir=img_file, files='data/train_files.txt', mode=args.mode)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = ImgDatasets(root_dir=img_file, files='data/test_files.txt', mode=args.mode, train=False)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # img, cond = next(iter(testloader))
    # from torchvision.utils import make_grid, save_image
    # save_image(make_grid(img, nrow=4), '../ori.png')
    # save_image(make_grid(cond, nrow=4), '../boxed.png')
    # assert False
    # Model
    print('Building model..')
    net = Glow(num_channels=args.num_channels,
               num_levels=args.num_levels,
               num_steps=args.num_steps,
               mode=args.mode)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    start_epoch = 0
    if args.resume !=None:
        # Load checkpoint.
        print('Resuming from '+args.resume)
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)

        net.load_state_dict(checkpoint['net'])
        global best_loss
        global global_step
        best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']
        global_step = start_epoch * len(trainset)

    loss_fn = util.NLLLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.warm_up))

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch, net, trainloader, device, optimizer, scheduler,
              loss_fn, args.max_grad_norm)
        test(epoch, net, testloader, device, loss_fn, args)


@torch.enable_grad()
def train(epoch, net, trainloader, device, optimizer, scheduler, loss_fn, max_grad_norm):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, cond_x in trainloader:
            x , cond_x= x.to(device), cond_x.to(device)
            optimizer.zero_grad()
            z, sldj = net(x, cond_x, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            if max_grad_norm > 0:
                util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()
            scheduler.step(global_step)

            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg),
                                     lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))
            global_step += x.size(0)


@torch.no_grad()
def sample(net, gray_img, device, sigma=0.6):
    B, C, W, H = gray_img.shape
    z = torch.randn((B, 3, 64, 64), dtype=torch.float32, device=device) * sigma
    x, _ = net(z, gray_img, reverse=True)
    x = torch.sigmoid(x)

    return x


@torch.no_grad()
def test(epoch, net, testloader, device, loss_fn, args):
    model_dir = os.path.join('ckpts', args.save_dir)
    sample_dir = os.path.join('samples', args.save_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()

    with tqdm(total=len(testloader.dataset)) as progress_bar:
        for x, x_cond in testloader:
            x, x_cond = x.to(device), x_cond.to(device)
            z, sldj = net(x, x_cond, reverse=False)
            loss = loss_fn(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            progress_bar.set_postfix(nll=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))

    # Save checkpoint
    # if loss_meter.avg < best_loss:
    print('Saving...')
    state = {
        'net': net.state_dict(),
        'test_loss': loss_meter.avg,
        'epoch': epoch,
        'parameter': [args.num_channels, args.num_levels, args.num_steps],
        'mode': args.mode
    }
    torch.save(state, os.path.join(model_dir, '{}_epoch.pth.tar'.format(epoch)))
    if loss_meter.avg < best_loss:
        print("===best score is: {:4f}".format(loss_meter.avg))
        best_loss = loss_meter.avg

    # origin_img, cond= next(iter(testloader))
    # origin_img, cond = origin_img.to(device), cond.to(device)

    # Sample
    images = sample(net, x_cond, device)
    img_list = []
    for i in range(x.size(0)):
        img_list.append(x[i].unsqueeze(0))
        img_list.append(x_cond[i].unsqueeze(0))
        img_list.append(images[i].unsqueeze(0))
    img_list = torch.cat(img_list, 0)
    images_concat = torchvision.utils.make_grid(img_list, nrow=3, padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, os.path.join(sample_dir, 'epoch_{}.png'.format(epoch)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glow on CelebA')

    def str2bool(s):
        return s.lower().startswith('t')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per GPU')
    parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
    parser.add_argument('--num_channels', '-C', default=128, type=int, help='Number of channels in hidden layers')
    parser.add_argument('--num_levels', '-L', default=6, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=16, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs to train')
    # parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--warm_up', default=500000, type=int, help='Number of steps for lr warm-up')
    parser.add_argument('--mode', default="random_inpainting", choices=['gray', 'sketch', 'random_inpainting', 'fix_inpainting'])
    parser.add_argument('--data_dir', default='/home/jkshark/data/celeba', type=str)
    parser.add_argument('--save_dir', default='random_inpainting', type=str)

    best_loss = float('inf')
    global_step = 0

    main(parser.parse_args())
