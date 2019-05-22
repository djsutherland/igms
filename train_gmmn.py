#!/usr/bin/env python
import argparse
import datetime
from functools import partial
import os
import shutil

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gmmn.datasets import get_dataset
from gmmn.featurize import load_featurizer
from gmmn.generator import make_generator
from gmmn.mmd import mmd2, Estimator
from gmmn.kernels import pick_kernel
from gmmn.utils import get_optimizer, pil, set_other_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir")
    parser.add_argument(
        "--force", action="store_true", help="Delete OUTDIR if it exists."
    )

    parser.add_argument("--out-size", type=int, default=64)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--dataset", default="celebA")
    parser.add_argument("--data-workers", type=int, default=2)
    parser.add_argument("--generator", default="dcgan")
    parser.add_argument("--optimizer", default="adam:1e-4")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--featurizer", default="through:smoothing")
    parser.add_argument("--kernel", default="linear")
    parser.add_argument("--mmd-estimator", default="unbiased")

    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled
    torch.backends.cudnn.benchmark = True

    if os.path.exists(args.outdir):
        if args.force:
            shutil.rmtree(args.outdir)
        else:
            try:
                os.rmdir(args.outdir)
            except OSError:
                parser.error(
                    f"{args.outdir} already exists; choose a different one or --force"
                )
    fn = partial(os.path.join, args.outdir)

    device = args.device
    generator = make_generator(
        args.generator, output_size=args.out_size, z_dim=args.z_dim
    ).to(device)
    opt = get_optimizer(args.optimizer)(generator.parameters())

    featurizer = load_featurizer(args.featurizer).to(device)
    kernel = pick_kernel(args.kernel)
    estimator = Estimator[args.mmd_estimator.upper()]

    dataset = get_dataset(args.dataset, out_size=args.out_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        # pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=args.data_workers,
        worker_init_fn=set_other_seeds,
    )

    def checkpoint(name):
        torch.save(
            {
                "args": args,
                "generator_state_dict": generator.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            },
            fn(f"checkpoint_{name}.pt"),
        )

    latents = torch.empty(args.batch_size, generator.z_dim, 1, 1, device=device)
    generator.train()

    os.makedirs(args.outdir)
    log_latents = torch.empty(64, generator.z_dim, 1, 1, device=device)
    log_latents.uniform_(-1, 1)
    pil(next(iter(dataloader))[0][: log_latents.shape[0]]).save(fn("real.jpg"))
    checkpoint(0)

    start = datetime.datetime.now()

    with open(fn("log.txt"), "w", buffering=1) as log_f:
        for epoch in tqdm(range(25), unit="epoch", dynamic_ncols=True):
            with tqdm(dataloader, unit="batch", dynamic_ncols=True) as bar:
                for batch_i, (real_imgs, real_ys) in enumerate(bar):
                    real_imgs = real_imgs.to(device)
                    real_noised = real_imgs + torch.randn_like(real_imgs) * 0.25
                    real_reps = featurizer(real_noised)

                    opt.zero_grad()
                    latents.uniform_(-1, 1)
                    fake_imgs = generator(latents)
                    fake_noised = fake_imgs + torch.randn_like(fake_imgs) * 0.25
                    fake_reps = featurizer(fake_noised)

                    loss = mmd2(kernel(real_reps, fake_reps), estimator=estimator)
                    loss.backward()
                    opt.step()

                    l_str = f"{loss.item():,.3f}"
                    bar.set_postfix(loss=l_str)

                    if batch_i % 100 == 0:
                        diff = datetime.datetime.now() - start
                        s = f"{epoch:>3} / {batch_i:>7,}: {l_str:>14}    ({diff})"
                        bar.write(s)
                        log_f.write(s + "\n")

                    if batch_i % 500 == 0:
                        pth = fn(f"fake_{epoch}_{batch_i}.jpg")
                        pil(generator(log_latents)).save(pth)
        checkpoint(epoch + 1)


if __name__ == "__main__":
    main()
