import os
import time
import random
import glob
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import load_image, save_image, total_variation_loss
from transformer_net import SimpleTransformerNet
from vgg import VGG16Features

# -----------------------------
# Simple Dataset
# -----------------------------
class ContentImageDataset(Dataset):
    def __init__(self, root, size):
        self.paths = glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True) + \
                     glob.glob(os.path.join(root, '**', '*.png'), recursive=True)
        self.tf = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8,1.0), ratio=(0.9,1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_image(self.paths[i])
        return self.tf(img), 0  # dummy label

# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Stable Style Transfer")
    p.add_argument('--dataset',      required=True)
    p.add_argument('--style-image',  required=True)
    p.add_argument('--demo-image',   required=True)
    p.add_argument('--save-dir',     default='saved_models')
    p.add_argument('--image-size',   type=int, default=256)
    p.add_argument('--batch-size',   type=int, default=4)
    p.add_argument('--epochs',       type=int, default=20)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--content-weight', type=float, default=1.0)
    p.add_argument('--style-weight',   type=float, default=0.1)
    p.add_argument('--tv-weight',      type=float, default=1e-6)
    p.add_argument('--max-images',     type=int, default=8000)
    p.add_argument('--cuda',           action='store_true')
    return p.parse_args()

# -----------------------------
# Main Training Loop
# -----------------------------
def main():
    args   = parse_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    print("Device:", device)

    # reproducibility
    random.seed(0)
    torch.manual_seed(0)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(0)

    # dataset & loader
    ds = ContentImageDataset(args.dataset, args.image_size)
    total = len(ds)
    n     = min(total, args.max_images)
    idx   = random.sample(range(total), n)
    loader = DataLoader(
        Subset(ds, idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )

    # fixed demo image
    demo = load_image(args.demo_image, size=args.image_size)
    demo = transforms.ToTensor()(demo).unsqueeze(0).to(device)

    # style statistics (mean & var)
    vgg = VGG16Features(requires_grad=False).to(device)
    style_img = load_image(args.style_image, size=args.image_size)
    style_t   = transforms.ToTensor()(style_img).unsqueeze(0).to(device)
    with torch.no_grad():
        fs = vgg(style_t)
    style_stats = {
        name: (
            getattr(fs, name).mean(dim=[2,3], keepdim=True),
            getattr(fs, name).var (dim=[2,3], keepdim=True, unbiased=False)
        )
        for name in fs._fields
    }

    # model, optimizer, scheduler, scaler
    model     = SimpleTransformerNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)
    scaler    = GradScaler()    # no args here
    l1        = nn.L1Loss()

    print("🚀 Starting training...")
    for ep in range(1, args.epochs+1):
        model.train()
        t0 = time.time()
        cum_c = cum_s = cum_tv = 0.0

        for i, (x, _) in enumerate(loader, 1):
            x = x.to(device)
            with autocast():   # no args here
                y = torch.clamp(model(x), 0, 1)
                fy, fx = vgg(y), vgg(x)

                # content loss (relu2_2)
                loss_c = l1(fy.relu2_2, fx.relu2_2)

                # style loss (mean + var)
                loss_s = 0.0
                for name, (mu_s, var_s) in style_stats.items():
                    f_y  = getattr(fy, name)
                    mu_y = f_y.mean([2,3], keepdim=True)
                    var_y= f_y.var ([2,3], keepdim=True, unbiased=False)
                    loss_s += l1(mu_y, mu_s) + l1(var_y, var_s)

                # total variation loss
                loss_tv = total_variation_loss(y, args.tv_weight)

                total_loss = (
                    args.content_weight * loss_c +
                    args.style_weight   * loss_s +
                    loss_tv
                )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            cum_c  += loss_c.item()
            cum_s  += loss_s.item()
            cum_tv += loss_tv.item()

        # save demo stylization
        with torch.no_grad():
            out = torch.clamp(model(demo), 0, 1).cpu()
            save_image(out, f"checkpoints/demo_ep{ep:02d}.png")

        # save model checkpoint
        torch.save(
            model.state_dict(),
            os.path.join(args.save_dir, f"epoch{ep:02d}.pth")
        )

        print(
            f"Epoch {ep}/{args.epochs} | "
            f"time={time.time()-t0:.1f}s | "
            f"C={cum_c/len(loader):.3f} | "
            f"S={cum_s/len(loader):.3f} | "
            f"TV={cum_tv/len(loader):.3f} | "
            f"LR={scheduler.get_last_lr()[0]:.2e}"
        )

    # final save
    torch.save(model.state_dict(), os.path.join(args.save_dir, "final.pth"))
    print("🏁 Training complete!")

if __name__ == '__main__':
    main()
