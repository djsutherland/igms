#!/usr/bin/env python
import datetime
from functools import partial
from glob import glob
import json
import os
import re
import shlex
import shutil
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from igms.datasets import get_dataset
from igms.featurize import load_featurizer
from igms.generator import make_generator
from igms.mmd import mmd2, Estimator
from igms.kernels import pick_kernel
from igms.utils import ArgumentParser, get_optimizer, pil, set_other_seeds


def parse_args(argv=None, check_dir=True):
    parser = ArgumentParser()
    parser.add_argument("outdir")
    parser.add_argument(
        "--force", action="store_true", help="Delete OUTDIR if it exists."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume the last checkpoint from OUTDIR. "
            "Overrides all other arguments except DEVICE and DATA_WORKERS."
        ),
    )

    ds = parser.add_argument_group("dataset options")
    ds.add_argument("--dataset", default="celebA")
    ds.add_argument("--out-size", type=int, default=64)

    gen = parser.add_argument_group("generator options")
    gen.add_argument("--z-dim", type=int, default=128)
    gen.add_argument("--batch-size", type=int, default=128)
    gen.add_argument("--generator", default="dcgan")

    opt = parser.add_argument_group("optimizer options")
    opt.add_argument("--optimizer", default="adam:1e-4")
    opt.add_argument("--epochs", type=int, default=25)

    loss = parser.add_argument_group("loss options")
    loss.add_argument("--featurizer", default="through:smoothing")
    loss.add_argument("--image-noise", type=float, default=0)
    loss.add_argument("--kernel", default="linear")
    loss.add_argument(
        "--mmd-estimator",
        default="unbiased",
        type=lambda s: s.upper().replace("-", "_"),
    )

    comp = parser.add_argument_group("computation options")
    comp.add_argument("--data-workers", type=int, default=2)
    comp.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    if args.resume:
        checkpoints = glob(f"{args.outdir}/checkpoint_*.pt")
        if not checkpoints:
            parser.error(f"Couldn't find any checkpoints to resume in {args.outdir}")

        pat = re.compile(r"/checkpoint_(\d+)_(\d+)\.pt$")

        def checkpoint_idx(s):
            return tuple(int(x) for x in pat.search(s).groups())

        checkpoint = torch.load(max(checkpoints, key=checkpoint_idx))
        checkpoint_args = checkpoint["args"]
        checkpoint_args.device = args.device
        checkpoint_args.data_workers = args.data_workers
        checkpoint_args.resume = True
        return checkpoint_args, checkpoint
    elif check_dir:
        if os.path.exists(args.outdir):
            ls = set(os.listdir(args.outdir))
            if args.force or ls.issubset({"args.json", "argv.txt"}):
                shutil.rmtree(args.outdir)
            else:
                parser.error(
                    f"{args.outdir} already exists; "
                    "use --resume, or --force to delete"
                )
        os.makedirs(args.outdir)

        v = parser._read_args_from_files(sys.argv[1:] if argv is None else argv)
        with open(os.path.join(args.outdir, "argv.txt"), "x") as f:
            print(
                "# This file has the effective arguments passed, one per line.\n"
                "# Note there might be duplicates (later taking precedence).\n"
                "# You can run them again with "
                f"`{os.path.basename(__file__)} @path/to/argv.txt`.\n"
                "# See args.json for how they were parsed by argparse.\n",
                file=f,
            )
            for arg in v:
                print(shlex.quote(arg), file=f)
        with open(os.path.join(args.outdir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    return args, {}


def main():
    args, checkpoint = parse_args()

    assert torch.cuda.is_available()
    assert torch.backends.cudnn.enabled
    torch.backends.cudnn.benchmark = True

    generator = make_generator(
        args.generator, output_size=args.out_size, z_dim=args.z_dim
    ).to(args.device)
    opt = get_optimizer(args.optimizer)(generator.parameters())

    if checkpoint:
        generator.load_state_dict(checkpoint["generator_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        log_latents = checkpoint["log_latents"].to(args.device)
        resume = (checkpoint["epoch"], checkpoint["batches"])
    else:
        log_latents = torch.empty(64, generator.z_dim, device=args.device)
        log_latents.uniform_(-1, 1)
        resume = None

    featurizer = load_featurizer(args.featurizer).to(args.device)
    top_kernel = pick_kernel(args.kernel)
    estimator = Estimator[args.mmd_estimator]

    if args.image_noise:
        real_noise = torch.empty(
            args.batch_size, *generator.output_shape, device=args.device
        )
        fake_noise = torch.empty(
            args.batch_size, *generator.output_shape, device=args.device
        )

    def loss_fn(real_imgs, fake_imgs):
        if args.image_noise:
            real_imgs = real_imgs + real_noise.normal_(std=args.image_noise)
            fake_imgs = fake_imgs + fake_noise.normal_(std=args.image_noise)
        real_reps = featurizer(real_imgs)
        fake_reps = featurizer(fake_imgs)
        return mmd2(top_kernel(real_reps, fake_reps), estimator=estimator)

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

    train_loop(args, dataloader, generator, loss_fn, opt, log_latents, resume)


def train_loop(args, dataloader, generator, loss_fn, opt, log_latents, resume=None):
    fn = partial(os.path.join, args.outdir)
    device = args.device

    latents = torch.empty(args.batch_size, generator.z_dim, device=device)
    generator.train()

    with open(fn("log.txt"), "a", buffering=1) as log_f:

        def write_line(line):
            s = f"[{datetime.datetime.now():%Y-%m-%d %H:%m:%S}]  {line}"
            tqdm.write(s)
            log_f.write(s + "\n")

        def checkpoint(epoch, batches=0):
            d = {
                "args": args,
                "generator_state_dict": generator.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "log_latents": log_latents,
                "epoch": epoch,
                "batches": batches,
            }

            name = orig_name = f"checkpoint_{epoch}_{batches}.pt"
            while os.path.exists(fn(name)):
                name += "~"
            torch.save(d, fn(name))

            if name != orig_name:
                write_line(
                    f"ERROR: checkpoint {orig_name} already exists! "
                    f"Saved to {name} instead, now aborting."
                )
                sys.exit(3)
            else:
                write_line(f"saved checkpoint: {name}")

        def save_samples(name):
            generator.eval()
            file = f"fake_{name}.jpg"
            pil(generator(log_latents)).save(fn(file))
            generator.train()
            write_line(f"saved samples: {file}")

        if resume:
            resume_epoch, resume_batch = resume
            epochs = range(resume_epoch, args.epochs)
            skip_batches = resume_batch + 1
            write_line(f"Resuming from {resume_epoch} / {resume_batch}")
        else:
            pil(next(iter(dataloader))[0][: log_latents.shape[0]]).save(fn("real.jpg"))
            checkpoint(0)
            epochs = range(args.epochs)
            skip_batches = 0

        for epoch in tqdm(epochs, unit="epoch", dynamic_ncols=True):
            with tqdm(dataloader, unit="batch", dynamic_ncols=True) as bar:
                for batch_i, (real_imgs, real_ys) in enumerate(bar):
                    if skip_batches:
                        skip_batches -= 1
                        continue

                    real_imgs = real_imgs.to(device)

                    opt.zero_grad()
                    latents.uniform_(-1, 1)
                    fake_imgs = generator(latents)

                    loss = loss_fn(real_imgs, fake_imgs)
                    loss.backward()
                    opt.step()

                    l_str = f"{loss.item():,.3f}"
                    bar.set_postfix(loss="=" + l_str.rjust(11))

                    if batch_i % 100 == 0:
                        write_line(f"{epoch:>3} / {batch_i:>7,}: {l_str:>14}")

                    if batch_i % 500 == 0:
                        save_samples(f"{epoch}_{batch_i}")
            checkpoint(epoch, batch_i)

        write_line(f"{epoch:>3} / {batch_i:>7,}: {l_str:>14}")
        save_samples(f"{epoch}_{batch_i}")


if __name__ == "__main__":
    main()
