'''
Author: Sunhr
LastEditors: Sunhr
Description: file content
'''
import argparse
import os
import torch
from typing import Optional, Sequence

from .trainer import InverseScatteringTrainer, TrainerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepCSI inverse scattering trainer")
    parser.add_argument("--expname", required=True, help="Name of the experiment directory")
    parser.add_argument("--basedir", default="./results", help="Directory to store outputs")
    parser.add_argument("--params_path", required=True, help="Ground-truth permittivity map (.npy)")
    parser.add_argument("--recdata_path", required=True, help="Measured scattered field (.npy or Fresnel .exp)")
    parser.add_argument(
        "--method",
        default="fd-isp",
        choices=["fd-isp", "pdtot-isp"],
        help="Inversion objective",
    )
    parser.add_argument("--freq", required=True, help="Comma-separated GHz frequencies, e.g., 3,4,5,")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--L_doi", type=float, default=0.3)
    parser.add_argument("--R_t", type=float, default=2.0)
    parser.add_argument("--R_r", type=float, default=2.2)
    parser.add_argument("--N_rec", type=int, default=360)
    parser.add_argument("--N_inc", type=int, default=16)
    parser.add_argument("--grid_num", type=int, default=96)
    parser.add_argument("--max_iter", type=int, default=3000)
    parser.add_argument("--netdepth", type=int, default=12)
    parser.add_argument("--netwidth", type=int, default=128)
    parser.add_argument("--multires", type=int, default=10)
    parser.add_argument("--lrate", type=float, default=5e-2)
    parser.add_argument("--lrate_decay", type=float, default=4.0)
    parser.add_argument("--params_lrate", type=float, default=5e-2)
    parser.add_argument("--params_lrate_decay", type=float, default=2.0)
    parser.add_argument("--max_params", type=float, default=3.0)
    parser.add_argument("--regularizer_weight", type=float, default=0.1)
    parser.add_argument("--regularizer_decay", type=float, default=1.0)
    parser.add_argument("--regularizer", choices=["tv_l1", "tv_l2", "mrtv"], default="tv_l1")
    parser.add_argument("--i_regularizer", type=int, default=1)
    parser.add_argument("--params_constraint", action="store_true")
    parser.add_argument("--noise_ratio", type=float, default=0.0)
    parser.add_argument("--result_file", default="", help="Optional CSV file to append metrics")
    parser.add_argument("--save_metric", action="store_true", help="Compute SSIM (requires torchmetrics)")
    parser.add_argument("--i_print", type=int, default=50)
    parser.add_argument("--i_testset", type=int, default=500)
    parser.add_argument("--i_weights", type=int, default=1000)
    parser.add_argument("--J_network", choices=["single-mlp", "multi-mlp"], default="single-mlp")
    return parser


def create_config(namespace: argparse.Namespace) -> TrainerConfig:
    kwargs = {k: getattr(namespace, k) for k in TrainerConfig.__annotations__.keys()}
    kwargs["basedir"] = os.path.abspath(kwargs["basedir"])
    if 'TwinDiel' in kwargs['recdata_path']:
        kwargs['N_inc'] = 18
    return TrainerConfig(**kwargs)


def main(argv: Optional[Sequence[str]] = None) -> None:
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore[attr-defined]
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = create_config(args)
    trainer = InverseScatteringTrainer(cfg)
    trainer.train_model()


if __name__ == "__main__":
    main()
