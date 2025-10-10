#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import numpy as np
import torch

try:
    if __package__ in (None, ""):
        import sys

        ROOT = Path(__file__).resolve().parents[1]
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.trainer import InverseScatteringTrainer, TrainerConfig
    else:
        from .trainer import InverseScatteringTrainer, TrainerConfig
except ImportError as exc:  # pragma: no cover - defensive branch
    raise RuntimeError("Failed to import DeepCSI trainer dependencies") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic scattered field measurements")
    parser.add_argument("--expname", default="forward-sim", help="Temporary experiment name")
    parser.add_argument("--basedir", default="./scratch", help="Directory to hold intermediate tensors")
    parser.add_argument("--params_path", required=True, help="Permittivity map (.npy)")
    parser.add_argument("--output", required=True, help="Destination .npy file")
    parser.add_argument("--freq", required=True, help="Comma separated frequencies, e.g., 3,4,")
    parser.add_argument("--L_doi", type=float, default=0.3)
    parser.add_argument("--R_t", type=float, default=2.0)
    parser.add_argument("--R_r", type=float, default=2.2)
    parser.add_argument("--N_rec", type=int, default=360)
    parser.add_argument("--N_inc", type=int, default=16)
    parser.add_argument("--grid_num", type=int, default=96)
    parser.add_argument("--netdepth", type=int, default=1)
    parser.add_argument("--netwidth", type=int, default=8)
    parser.add_argument("--multires", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore[attr-defined]
    parser = build_parser()
    args = parser.parse_args()

    cfg = TrainerConfig(
        expname=args.expname,
        basedir=os.path.abspath(args.basedir),
        params_path=args.params_path,
        recdata_path="",
        method="fd-isp",
        freq=args.freq,
        seed=args.seed,
        L_doi=args.L_doi,
        R_t=args.R_t,
        R_r=args.R_r,
        N_rec=args.N_rec,
        N_inc=args.N_inc,
        grid_num=args.grid_num,
        max_iter=1,
        netdepth=max(args.netdepth, 1),
        netwidth=max(args.netwidth, 8),
        multires=max(args.multires, 1),
    )
    trainer = InverseScatteringTrainer(cfg)

    epsilon = torch.tensor(np.load(args.params_path), dtype=torch.float32, device=trainer.device)
    if epsilon.shape[0] != args.grid_num:
        epsilon = torch.nn.functional.interpolate(
            epsilon.unsqueeze(0).unsqueeze(0),
            size=(args.grid_num, args.grid_num),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    solution = trainer.forward_solution(epsilon)
    output = {
        "J_trad": solution["J"].cpu().numpy(),
        "E_sca": solution["E_sca"].cpu().numpy(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, output)
    print(f"Saved measurement to {args.output}")


if __name__ == "__main__":
    main()
