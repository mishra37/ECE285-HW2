"""Train a DCGAN on cached WikiArt images.

Usage: python train.py --data data/wikiart64.npy --epochs 50 --out outputs/run1
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import NpyImages
from models import Discriminator, Generator, weights_init


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="outputs/run1")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--label-smooth", type=float, default=0.9, help="target for real labels (1.0 = off)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(f"{args.out}/samples", exist_ok=True)
    with open(f"{args.out}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    device = get_device()
    print("device:", device)

    ds = NpyImages(args.data)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=args.workers, pin_memory=device.type == "cuda",
                    persistent_workers=args.workers > 0)
    print(f"{len(ds)} images, {len(dl)} iters/epoch")

    G = Generator(args.nz, args.ngf).to(device)
    D = Discriminator(args.ndf).to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, 0.999))
    bce = nn.BCEWithLogitsLoss()
    fixed_z = torch.randn(64, args.nz, 1, 1, device=device)

    history = {"g_loss": [], "d_loss": [], "d_real": [], "d_fake": []}
    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        optG.load_state_dict(ck["optG"]); optD.load_state_dict(ck["optD"])
        history, start_epoch, fixed_z = ck["history"], ck["epoch"] + 1, ck["fixed_z"].to(device)

    for epoch in range(start_epoch, args.epochs):
        G.train(); D.train()
        t0 = time.time()
        for i, real in enumerate(dl):
            real = real.to(device, non_blocking=True)
            b = real.size(0)
            z = torch.randn(b, args.nz, 1, 1, device=device)
            fake = G(z)

            # ---- D step: maximize log D(x) + log(1 - D(G(z)))
            optD.zero_grad(set_to_none=True)
            out_real = D(real)
            out_fake = D(fake.detach())
            d_loss = bce(out_real, torch.full_like(out_real, args.label_smooth)) + \
                     bce(out_fake, torch.zeros_like(out_fake))
            d_loss.backward()
            optD.step()

            # ---- G step: non-saturating loss, maximize log D(G(z))
            optG.zero_grad(set_to_none=True)
            out = D(fake)
            g_loss = bce(out, torch.ones_like(out))
            g_loss.backward()
            optG.step()

            history["g_loss"].append(g_loss.item())
            history["d_loss"].append(d_loss.item())
            history["d_real"].append(torch.sigmoid(out_real).mean().item())
            history["d_fake"].append(torch.sigmoid(out_fake).mean().item())
            if i % 100 == 0:
                print(f"[{epoch+1}/{args.epochs}][{i}/{len(dl)}] D {d_loss.item():.3f} G {g_loss.item():.3f} "
                      f"D(x) {history['d_real'][-1]:.3f} D(G(z)) {history['d_fake'][-1]:.3f}")

        G.eval()
        with torch.no_grad():
            save_image(G(fixed_z), f"{args.out}/samples/epoch_{epoch+1:03d}.png", nrow=8, normalize=True, value_range=(-1, 1))
        ck = {"G": G.state_dict(), "D": D.state_dict(), "optG": optG.state_dict(), "optD": optD.state_dict(),
              "history": history, "epoch": epoch, "fixed_z": fixed_z.cpu(), "args": vars(args)}
        torch.save(ck, f"{args.out}/last.pt")
        if (epoch + 1) % 10 == 0:
            torch.save(G.state_dict(), f"{args.out}/G_epoch{epoch+1:03d}.pt")
        plot_losses(history, len(dl), f"{args.out}/loss_curves.png")
        print(f"epoch {epoch+1} done in {time.time()-t0:.0f}s")

    torch.save(G.state_dict(), f"{args.out}/G_final.pt")
    with open(f"{args.out}/history.json", "w") as f:
        json.dump(history, f)


def smooth(x, k=50):
    if len(x) < k:
        return x
    import numpy as np
    return np.convolve(x, np.ones(k) / k, mode="valid")


def plot_losses(h, iters_per_epoch, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for key, lbl in [("d_loss", "Discriminator"), ("g_loss", "Generator")]:
        ax[0].plot(h[key], alpha=0.2)
        ax[0].plot(range(49, 49 + len(smooth(h[key]))) if len(h[key]) >= 50 else range(len(h[key])),
                   smooth(h[key]), label=lbl, color=ax[0].lines[-1].get_color())
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Training losses")
    ax[1].plot(smooth(h["d_real"]), label="D(x)")
    ax[1].plot(smooth(h["d_fake"]), label="D(G(z))")
    ax[1].set_xlabel("iteration"); ax[1].legend(); ax[1].set_title("Discriminator outputs")
    for a in ax:
        sec = a.secondary_xaxis("top", functions=(lambda x: x / iters_per_epoch, lambda e: e * iters_per_epoch))
        sec.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
