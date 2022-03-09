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
from torch.autograd import Variable
from models import Glow
from models.discriminator import Discriminator, LS_Discriminator
from tqdm import tqdm


def main(args):
    # Set up main device and scale batch size
    device = 'cuda' if torch.cuda.is_available() and args.gpu_ids else 'cpu'
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trainset = ImgDatasets(root_dir='/home/jkshark/data/celeba/images', files='data/train_files.txt', mode=args.mode)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = ImgDatasets(root_dir='/home/jkshark/data/celeba/images', files='data/test_files.txt', mode=args.mode, train=False)
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
    D_net = Discriminator(args.image_size).to(device)

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
    d_optimizer = optim.Adam(D_net.parameters(), lr=args.lr)
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.warm_up))

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train_w_D_W(epoch, net, D_net, trainloader, device, optimizer, d_optimizer, scheduler,
              loss_fn, args.max_grad_norm)
        test(epoch, net, D_net, testloader, device, loss_fn, args.mode)


def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)

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

@torch.enable_grad()
def train_w_D_W(epoch, net, D_net, trainloader, device, optimizer, d_optimizer, scheduler, loss_fn, max_grad_norm):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    flow_loss_meter = util.AverageMeter()
    D_loss_meter = util.AverageMeter()

    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, cond_x in trainloader:
            x , cond_x= x.to(device), cond_x.to(device)
            z, sldj = net(x, cond_x, reverse=False)
            # Compute loss with Fake
            with torch.no_grad():
                rec, _ = net(z, cond_x, reverse=True)

            #============ Train Flow model==========
            flow_loss = loss_fn(z, sldj)
            out_src = D_net(rec)
            flow_loss_rec = -torch.mean(out_src)
            loss = flow_loss + flow_loss_rec
            flow_loss_meter.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            # ========Use Discriminator==========
            #Compute loss with Real
            out_src = D_net(x)
            d_loss_real = -torch.mean(out_src)

            #COmpute loss with Fake
            out_src = D_net(rec)
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty
            alpha = torch.rand(x.size(0), 1, 1, 1).to(device)
            x_hat = (alpha * x.data + (1 - alpha) * rec.data).requires_grad_(True)
            out_src = D_net(x_hat)
            d_loss_gp = gradient_penalty(out_src, x_hat, device)

            #Train D_net
            lambda_gp = 10
            d_loss = d_loss_real + d_loss_fake + lambda_gp * d_loss_gp
            D_loss_meter.update(d_loss.item(), x.size(0))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            scheduler.step(global_step)
            progress_bar.set_postfix(nll=flow_loss_meter.avg,
                                     D_loss= D_loss_meter.avg,
                                     bpd=util.bits_per_dim(x, flow_loss_meter.avg),
                                     lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))
            global_step += x.size(0)

def train_w_D_lsGAN(epoch, net, D_net, trainloader, device, optimizer, d_optimizer, scheduler, loss_fn, max_grad_norm):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    flow_loss_meter = util.AverageMeter()
    D_loss_meter = util.AverageMeter()

    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, cond_x in trainloader:
            x , cond_x= x.to(device), cond_x.to(device)
            valid = Variable(torch.Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(torch.Tensor(x.shape[0], 1).fill_(0.0), requires_grad=False).to(device)

            z, sldj = net(x, cond_x, reverse=False)
            # Compute loss with Fake
            with torch.no_grad():
                rec, _ = net(z, cond_x, reverse=True)

            #============ Train Flow model==========
            flow_loss = loss_fn(z, sldj)
            out_src = D_net(rec)
            flow_loss_rec = torch.nn.MSELoss()(out_src, valid)
            loss = flow_loss + flow_loss_rec
            flow_loss_meter.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()

            # ========Use Discriminator==========
            # Compute loss with Real
            out_src = D_net(x)
            d_loss_real = torch.nn.MSELoss()(out_src, valid)

            out_src = D_net(rec)
            d_loss_fake = torch.nn.MSELoss()(out_src, fake)

            # Train D_net
            d_loss = d_loss_real + d_loss_fake
            D_loss_meter.update(d_loss.item(), x.size(0))
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            scheduler.step(global_step)
            progress_bar.set_postfix(nll=flow_loss_meter.avg,
                                     D_loss= D_loss_meter.avg,
                                     bpd=util.bits_per_dim(x, flow_loss_meter.avg),
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
def test(epoch, net, D_net, testloader, device, loss_fn, mode='color'):
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
    if loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'flow_net': net.state_dict(),
            'D_net': D_net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs('ckpts/inpainting_D_W', exist_ok=True)
        torch.save(state, 'ckpts/inpainting_D_W/best.pth.tar')
        best_loss = loss_meter.avg
    origin_img, cond = next(iter(testloader))
    origin_img, cond = origin_img.to(device), cond.to(device)

    # Sample
    images = sample(net, cond, device)
    os.makedirs('samples', exist_ok=True)
    img_list = []
    for i in range(16):
        img_list.append(origin_img[i].unsqueeze(0))
        img_list.append(cond[i].unsqueeze(0))
        img_list.append(images[i].unsqueeze(0))
    img_list = torch.cat(img_list, 0)
    images_concat = torchvision.utils.make_grid(img_list,  nrow=3, padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, 'samples/inpainting_D_W/epoch_{}.png'.format(epoch))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Glow on CelebA')

    def str2bool(s):
        return s.lower().startswith('t')

    parser.add_argument('--image_size', default=64, type=int, help='Input image size')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size per GPU')
    parser.add_argument('--benchmark', type=str2bool, default=True, help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default=[0,1], type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=-1., help='Max gradient norm for clipping')
    parser.add_argument('--num_channels', '-C', default=128, type=int, help='Number of channels in hidden layers')
    parser.add_argument('--num_levels', '-L', default=3, type=int, help='Number of levels in the Glow model')
    parser.add_argument('--num_steps', '-K', default=8, type=int, help='Number of steps of flow in each level')
    parser.add_argument('--num_epochs', default=500, type=int, help='Number of epochs to train')
    # parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--warm_up', default=500000, type=int, help='Number of steps for lr warm-up')
    parser.add_argument('--mode', default="random_inpainting", choices=['gray', 'sketch', 'random_inpainting', 'fix_inpainting'])
    best_loss = float('inf')
    global_step = 0

    main(parser.parse_args())
