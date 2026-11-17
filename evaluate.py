"""Compute Inception Score and FID with PyTorch-Ignite, following
https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/

Usage: python evaluate.py --data data/wikiart64.npy --ckpt outputs/run1/G_final.pt --n 10000
"""
import argparse
import json

import torch
import torch.nn.functional as F
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from data import NpyImages
from models import Generator


def interpolate(batch):
    """[-1,1] 64x64 -> [0,1] 299x299 for Inception-v3."""
    batch = (batch.clamp(-1, 1) + 1) / 2
    return F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=10000, help="number of real/fake images to compare")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--nz", type=int, default=100)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--out", default=None, help="json file to write scores to")
    p.add_argument("--samples", default=None, help="save a grid of 64 samples here")
    args = p.parse_args()

    # Ignite's FID accumulates in float64, which MPS doesn't support -> use CUDA or CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    G = Generator(args.nz, args.ngf).to(device)
    G.load_state_dict(torch.load(args.ckpt, map_location=device))
    G.eval()

    ds = NpyImages(args.data, hflip=False)
    idx = torch.randperm(len(ds))[: args.n]
    dl = DataLoader(Subset(ds, idx.tolist()), batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Blog default: FID on Inception softmax outputs (1000-d). Values are much smaller
    # than the "standard" FID from papers, which uses 2048-d pool3 features.
    fid_metric = FID(device=device)
    fid_pool_metric = FID(num_features=2048, device=device,
                          feature_extractor=InceptionModel(return_features=True, device=device))
    is_metric = InceptionScore(device=device, output_transform=lambda x: x[0])

    def evaluation_step(engine, batch):
        with torch.no_grad():
            z = torch.randn(batch.size(0), args.nz, 1, 1, device=device)
            fake = interpolate(G(z))
            real = interpolate(batch.to(device))
            return fake, real

    evaluator = Engine(evaluation_step)
    fid_metric.attach(evaluator, "fid")
    fid_pool_metric.attach(evaluator, "fid_pool3")
    is_metric.attach(evaluator, "is")
    evaluator.run(dl, max_epochs=1)
    m = evaluator.state.metrics
    res = {"fid": float(m["fid"]), "fid_pool3": float(m["fid_pool3"]), "is": float(m["is"]), "n_images": len(dl) * args.batch_size}
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
    if args.samples:
        with torch.no_grad():
            save_image(G(torch.randn(64, args.nz, 1, 1, device=device)), args.samples,
                       nrow=8, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    main()
