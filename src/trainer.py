import os
from dataclasses import dataclass
from time import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.special as sp
import torch.nn.functional as F
from tqdm import trange

from .models import NeJF, get_embedder, img2mse
from .utils import (
    LossRecorder,
    TotalVariationL1,
    TotalVariationL2,
    MultiDirectionalTV,
    besselh,
    epsilon_control,
    fresnel_data_preprocess,
    plot_J_figure_multifreqs,
    plot_params_figure,
    read_exp_data,
    save_results_row,
)


@dataclass
class TrainerConfig:
    expname: str
    basedir: str
    params_path: str
    recdata_path: str
    method: str
    freq: str
    seed: int = 0
    L_doi: float = 0.3
    R_t: float = 2.0
    R_r: float = 2.2
    N_rec: int = 360
    N_inc: int = 16
    grid_num: int = 96
    max_iter: int = 3000
    netdepth: int = 12
    netwidth: int = 128
    multires: int = 10
    lrate: float = 5e-2
    lrate_decay: float = 4.0
    params_lrate: float = 5e-2
    params_lrate_decay: float = 2.0
    max_params: float = 3.0
    regularizer_weight: float = 0.1
    regularizer_decay: float = 1.0
    regularizer: str = "tv_l1"
    i_regularizer: int = 1
    params_constraint: bool = False
    noise_ratio: float = 0.0
    result_file: str = ""
    save_metric: bool = False
    i_print: int = 50
    i_testset: int = 500
    i_weights: int = 1000
    J_network: str = "single-mlp"


class InverseScatteringTrainer(nn.Module):
    def __init__(self, cfg: TrainerConfig) -> None:
        super().__init__()
        self.args = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.exp_dir = os.path.join(cfg.basedir, cfg.expname)
        os.makedirs(self.exp_dir, exist_ok=True)

        self.freqs = torch.tensor(
            [float(f) for f in cfg.freq.split(",") if f]
        )
        if self.freqs.numel() == 0:
            raise ValueError("At least one frequency must be provided")
        self.num_freqs = int(self.freqs.numel())
        c0 = 3e8
        lam0 = c0 / (self.freqs * 1e9)
        self.k0 = 2 * torch.pi / lam0
        self.a_eqv = np.sqrt((cfg.L_doi / (cfg.grid_num - 1)) ** 2 / np.pi)
        self.eta_0 = 120 * np.pi

        self.step_size = cfg.L_doi / (cfg.grid_num - 1)
        self.cell_area = self.step_size ** 2
        self.N_cell = cfg.grid_num ** 2

        self.embed_fn, self.embed_dim = get_embedder(self.args.multires)
        self._assemble_green_tensors()
        self._build_models()

    # ------------------------------------------------------------------
    def _assemble_green_tensors(self) -> None:
        args = self.args
        grid = args.grid_num
        L = args.L_doi
        N_cell = self.N_cell

        # coords_lin = torch.linspace(-L / 2, L / 2, grid, device=self.device)
        # x_dom, y_dom = torch.meshgrid(coords_lin, coords_lin, indexing="xy")
        y_dom,x_dom = torch.meshgrid([torch.arange(L/2,-L/2-L/(grid-1)/2,-L/(grid-1)),
                                      torch.arange(-L/2,L/2+L/(grid-1)/2,L/(grid-1))])
        x_dom = x_dom.to(self.device)
        y_dom = y_dom.to(self.device)
        xy_dom = torch.stack([x_dom, y_dom], dim=-1)
        self.coords_eps = xy_dom

        theta_inc = torch.arange(0, 2*torch.pi, 2*torch.pi/args.N_inc)
        theta_rec = torch.arange(0, 2*torch.pi, 2*torch.pi/args.N_rec)
        xy_t = torch.cat([torch.cos(theta_inc).unsqueeze(-1)*args.R_t, torch.sin(theta_inc).unsqueeze(-1)*args.R_t], -1)
        xy_r = torch.cat([torch.cos(theta_rec).unsqueeze(-1)*args.R_r, torch.sin(theta_rec).unsqueeze(-1)*args.R_r], -1)
        xy_t = xy_t.to(self.device)
        xy_r = xy_r.to(self.device)
        self.coords_inc = torch.cat((torch.reshape(xy_dom.transpose(0, 1), [-1, 2]).unsqueeze(-2).repeat([1, args.N_inc, 1]),
                                    xy_t.unsqueeze(0).repeat([N_cell, 1, 1])), -1)
        if len(self.args.freq)>1 and self.args.J_network == 'single-mlp':
            self.coords_inc = torch.cat((self.k0.view(-1,1,1,1).repeat([1, N_cell, args.N_inc, 1]).to(self.device),self.coords_inc[None,...].repeat([self.num_freqs, 1,1,1])), -1)
        else:
            self.coords_inc = self.coords_inc[None,...]

        y_dom_flatten = y_dom.T.reshape([-1,1])
        x_dom_flatten = x_dom.T.reshape([-1,1])
        dist_cell = torch.sqrt((x_dom_flatten.repeat([1,N_cell])-x_dom_flatten.repeat([1,N_cell]).T)**2+
                        (y_dom_flatten.repeat([1,N_cell])-y_dom_flatten.repeat([1,N_cell]).T)**2)
        dist_cell = dist_cell + torch.eye(N_cell).to(self.device)

        kba = self.k0*self.a_eqv
        coeff = -1j * torch.pi * kba / 2
        
        Phi_mat = (coeff * sp.bessel_j1(kba)).view(-1,1,1) * besselh(0,2,self.k0.view(-1,1,1)*dist_cell[None,...])
        Phi_mat = Phi_mat*(torch.ones(N_cell)-torch.eye(N_cell))[None,...].to(self.device)
        self.Phi_mat = Phi_mat+((coeff*(besselh(1,2,self.k0*self.a_eqv))-1).view(-1,1,1)*torch.eye(N_cell)[None,...]).to(self.device)

        rho_mat_r = torch.sqrt((xy_r[:,0].unsqueeze(-1).repeat([1,N_cell])-x_dom_flatten.repeat([1,args.N_rec]).T)**2 
                        +(xy_r[:,1].unsqueeze(-1).repeat([1,N_cell])-y_dom_flatten.repeat([1,args.N_rec]).T)**2)
        self.Rec_mat = (coeff * sp.bessel_j1(kba)).view(-1,1,1) * besselh( 0, 2, self.k0.view(-1,1,1)*rho_mat_r[None,...])

        rho_mat_t = torch.sqrt((xy_t[:,0].unsqueeze(-1).repeat([1,N_cell])-x_dom_flatten.repeat([1,args.N_inc]).T)**2 
                            +(xy_t[:,1].unsqueeze(-1).repeat([1,N_cell])-y_dom_flatten.repeat([1,args.N_inc]).T)**2)
        T_mat = 1/(4*1j)*besselh(0,2,self.k0.view(-1,1,1)*rho_mat_t[None,...])
        self.E_inc = T_mat.transpose(1, 2)

        rho_mat_t_r = torch.sqrt((xy_r[:,0].unsqueeze(-1).repeat([1,args.N_inc])-xy_t[:,0].unsqueeze(-1).repeat([1,args.N_rec]).T)**2
                                     +(xy_r[:,1].unsqueeze(-1).repeat([1,args.N_inc])-xy_t[:,1].unsqueeze(-1).repeat([1,args.N_rec]).T)**2)
        self.E_inc_tr = 1/(4*1j)*besselh(0,2,self.k0.view(-1,1,1)*rho_mat_t_r[None,...])

    # ------------------------------------------------------------------
    def _build_models(self) -> None:
        skips = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
        if self.num_freqs > 1:
            input_ch = int(self.embed_dim * 2)
            nets = [
                NeJF(
                    D=self.args.netdepth,
                    W=self.args.netwidth,
                    input_ch=input_ch,
                    output_ch=2,
                    skips=skips,
                    tanh=True,
                ).to(self.device)
                for _ in range(self.num_freqs)
            ]
        else:
            input_ch = int(self.embed_dim * 2)
            nets = [
                NeJF(
                    D=self.args.netdepth,
                    W=self.args.netwidth,
                    input_ch=input_ch,
                    output_ch=2,
                    skips=skips,
                    tanh=True,
                ).to(self.device)
            ]
        self.model_J = nn.ModuleList(nets)

    # ------------------------------------------------------------------
    def _predict_currents(self) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        features = self.embed_fn(self.coords_inc.reshape(-1, self.coords_inc.shape[-1]))  # type: ignore[misc]
        for freq_idx in range(self.num_freqs):
            pred = self.model_J[freq_idx](features)
            outputs.append(pred.view(self.N_cell, self.args.N_inc, 2))
        stacked = torch.stack(outputs, dim=0)
        return torch.complex(stacked[..., 0], stacked[..., 1])

    def render(self, epsilon_param: torch.Tensor) -> Dict[str, torch.Tensor]:
        epsilon = epsilon_control(epsilon_param, self.args.params_constraint)
        xi_all = ((epsilon - 1)[None,...]*(1+0j)).repeat(self.num_freqs,1,1)
        xi_forward = torch.reshape(xi_all.transpose(1,2), [self.num_freqs, -1, 1])
        xi_forward_mat = torch.diag_embed(xi_forward.squeeze(-1))
        xi_E_inc = xi_forward_mat @ self.E_inc

        currents = self._predict_currents()
        if self.Rec_mat.dim() == 3:
            Esca = torch.matmul(self.Rec_mat, currents)
        else:
            Esca = torch.matmul(self.Rec_mat, currents.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)
        Phi_currents = torch.matmul(self.Phi_mat, currents)
        J_state = xi_E_inc + xi_forward_mat @ Phi_currents
        norm_inc = torch.mean(xi_E_inc.real ** 2 + xi_E_inc.imag ** 2)
        return {
            "epsilon": epsilon,
            "currents": currents,
            "Esca": Esca,
            "J_state": J_state,
            "xi_E_inc": xi_E_inc,
            "norm_inc": norm_inc,
        }

    # ------------------------------------------------------------------
    def _load_measurements(self) -> torch.Tensor:
        if self.args.recdata_path.lower().endswith(".npy"):
            data = np.load(self.args.recdata_path, allow_pickle=True).item()
            return torch.tensor(data["E_sca"], dtype=torch.complex64, device=self.device)
        return self._load_fresnel_measurements()

    def _load_fresnel_measurements(self) -> torch.Tensor:
        raw = read_exp_data(self.args.recdata_path, nf=9, ns=self.args.N_inc, nr=241).to(self.device)
        freq_indices = [int(float(f) - 2) for f in self.args.freq.split(",") if f]
        measured = torch.stack([raw[idx] for idx in freq_indices], dim=0)
        masks = fresnel_data_preprocess(self.num_freqs, 241, self.args.N_rec, self.args.N_inc, self.device)
        masks_exp = masks.unsqueeze(0).repeat(self.num_freqs, 1, 1, 1).to(torch.complex64)
        rec_mat = torch.matmul(
            masks_exp,
            self.Rec_mat.unsqueeze(1).repeat(1, self.args.N_inc, 1, 1),
        )
        self.Rec_mat = rec_mat  # shape: (freq, N_inc, 241, N_cell)
        E_inc_tr = torch.nan_to_num(self.E_inc_tr).permute(0, 2, 1).unsqueeze(-1)
        self.E_inc_tr = torch.matmul(masks_exp, E_inc_tr).squeeze(-1).permute(0, 2, 1)
        return measured.permute(0, 2, 1)

    # ------------------------------------------------------------------
    def train_model(self) -> None:
        epsilon_gt = torch.tensor(np.load(self.args.params_path), dtype=torch.float32, device=self.device)
        if epsilon_gt.shape[0] != self.args.grid_num:
            epsilon_gt = F.interpolate(
                epsilon_gt.unsqueeze(0).unsqueeze(0),
                size=(self.args.grid_num, self.args.grid_num),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        if 'fresnel' not in self.args.params_path.lower():
            epsilon_gt = epsilon_gt.T
        epsilon_param = nn.Parameter(torch.ones_like(epsilon_gt), requires_grad=True)

        optimizer_model = torch.optim.Adam(self.model_J.parameters(), lr=self.args.lrate, betas=(0.9, 0.999))
        optimizer_params = torch.optim.Adam([epsilon_param], lr=self.args.params_lrate, betas=(0.9, 0.999))
        recorder = LossRecorder(self.exp_dir, ["J_state_loss", "Esca_loss", "TV_loss", "Total_loss"])
        regularizers = {
            "tv_l1": TotalVariationL1(),
            "tv_l2": TotalVariationL2(),
            "mrtv": MultiDirectionalTV(),
        }
        regularizer_name = self.args.regularizer.lower()
        self.regular_item = regularizers.get(regularizer_name, TotalVariationL1())
        last_regular_loss = 0.0
        last_fd = 1e-6 * self.cell_area ** 2
        decay_begin = 500
        if self.args.max_iter > decay_begin and self.args.regularizer_decay < 1.0:
            delta_regularizer = (1 - self.args.regularizer_decay) / max(1, (self.args.max_iter - decay_begin) // 100)
        else:
            delta_regularizer = 0.0

        measurements = self._load_measurements()
        if self.args.noise_ratio > 0:
            noise = torch.randn_like(measurements) * self.args.noise_ratio * torch.mean(torch.abs(measurements)) / np.sqrt(2)
            measurements = measurements + noise

        try:
            from torchmetrics.image import StructuralSimilarityIndexMeasure  # type: ignore

            ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        except Exception:
            ssim_metric = None

        total_time = 0.0
        pbar = trange(self.args.max_iter, desc="Training", leave=True)
        for step in pbar:
            iter_start = time()
            optimizer_model.zero_grad()
            optimizer_params.zero_grad()

            render_out = self.render(epsilon_param)
            Esca = render_out["Esca"]
            currents = render_out["currents"]
            J_state = render_out["J_state"]
            norm_inc = render_out["norm_inc"] + torch.finfo(torch.float32).eps

            if self.args.method == "fd-isp":
                Esca_loss = (
                    img2mse(Esca.real, measurements.real) + img2mse(Esca.imag, measurements.imag)
                ) / torch.mean(measurements.real ** 2 + measurements.imag ** 2)
            elif self.args.method == "pdtot-isp":
                pred_total = torch.abs(Esca + self.E_inc_tr)
                true_total = torch.abs(measurements + self.E_inc_tr)
                Esca_loss = 2 * img2mse(pred_total, true_total) / torch.mean(torch.abs(measurements) ** 2)
            else:
                raise ValueError("Unsupported method: choose fd-isp or pdtot-isp")

            J_state_loss = (
                img2mse(J_state.real, currents.real) + img2mse(J_state.imag, currents.imag)
            ) / norm_inc
            regular_loss = self.regular_item(render_out["epsilon"])
            tv_loss = regular_loss
            if regularizer_name == "mrtv":
                if step == 0:
                    last_regular_loss = regular_loss.item()
                    last_fd = J_state_loss.item() * self.cell_area ** 2
                total_loss = (Esca_loss + J_state_loss) * (
                    (regular_loss + last_fd) / (last_regular_loss + last_fd)
                )
                if step % max(1, self.args.i_regularizer) == 0:
                    last_regular_loss = regular_loss.item()
                    last_fd = J_state_loss.item() * self.cell_area ** 2
            elif regularizer_name in {"tv_l1", "tv_l2"}:
                total_loss = Esca_loss + J_state_loss + self.args.regularizer_weight * regular_loss
            else:
                total_loss = Esca_loss + J_state_loss
            total_loss.backward()
            optimizer_model.step()
            optimizer_params.step()

            iter_time = time() - iter_start
            total_time += iter_time

            decay_rate = 0.1
            lr = self.args.lrate * (decay_rate ** (step / (self.args.lrate_decay * 1000)))
            params_lr = self.args.params_lrate * (decay_rate ** (step / (self.args.params_lrate_decay * 1000)))
            for pg in optimizer_model.param_groups:
                pg["lr"] = lr
            for pg in optimizer_params.param_groups:
                pg["lr"] = params_lr

            if delta_regularizer > 0 and step > decay_begin and regularizer_name in {"tv_l1", "tv_l2"}:
                scale = 1 - ((step - decay_begin) // 100) * delta_regularizer
                scale = max(self.args.regularizer_decay, scale)
                self.args.regularizer_weight = max(
                    self.args.regularizer_decay,
                    self.args.regularizer_weight * scale,
                )

            recorder.update(
                {
                    "J_state_loss": J_state_loss.item(),
                    "Esca_loss": Esca_loss.item(),
                    "TV_loss": tv_loss.item(),
                    "Total_loss": total_loss.item(),
                }
            )
            if step % self.args.i_print == 0:
                pbar.set_postfix({
                    "data": Esca_loss.item(),
                    "state": J_state_loss.item(),
                    "tv": tv_loss.item(),
                })
            if step % self.args.i_testset == 0 and step > 0:
                self._save_snapshot(render_out, step)

        recorder.plot_losses()
        recorder.save_history()

        final_render = self.render(epsilon_param)
        mse_val = torch.sqrt(F.mse_loss(final_render["epsilon"], epsilon_gt) / torch.mean(epsilon_gt ** 2)).item()
        data_misfit = torch.sqrt(
            img2mse(final_render["Esca"].real, measurements.real) + img2mse(final_render["Esca"].imag, measurements.imag)
        ).item()
        if ssim_metric is not None:
            ssim_val = ssim_metric(
                final_render["epsilon"].unsqueeze(0).unsqueeze(0),
                epsilon_gt.unsqueeze(0).unsqueeze(0),
            ).item()
        else:
            ssim_val = float("nan")

        if self.args.result_file:
            header = ["Experiment", "Method", "Freq", "Model Misfit", "Data Misfit", "SSIM", "Time (s)"]
            row = [
                self.args.expname,
                self.args.method,
                self.args.freq,
                mse_val,
                data_misfit,
                ssim_val,
                total_time,
            ]
            save_results_row(self.args.result_file, header, row)

    # ------------------------------------------------------------------

    def forward_solution(self, epsilon_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        epsilon = epsilon_map.to(self.device)
        chi = (epsilon - 1).to(torch.complex64).reshape(-1)
        xi_forward = chi.unsqueeze(0).repeat(self.num_freqs, 1)
        xi_forward_mat = torch.diag_embed(xi_forward)
        xi_E_inc = xi_forward_mat @ self.E_inc
        identity = torch.eye(self.N_cell, dtype=torch.complex64, device=self.device).unsqueeze(0).repeat(self.num_freqs, 1, 1)
        system = identity - xi_forward_mat @ self.Phi_mat
        J = torch.linalg.solve(system, xi_E_inc)
        if self.Rec_mat.dim() == 3:
            Esca = self.Rec_mat @ J
        else:
            Esca = torch.matmul(self.Rec_mat, J.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)
        return {"J": J, "E_sca": Esca}

    def _save_snapshot(self, render_out: Dict[str, torch.Tensor], step: int) -> None:
        save_base = os.path.join(self.exp_dir, f"testset_{step:06d}")
        currents = render_out["currents"].detach().cpu().numpy()
        Esca = render_out["Esca"].detach().cpu().numpy()
        epsilon = render_out["epsilon"].detach().cpu().numpy()
        np.save(save_base + ".npy", {"J_pred": currents, "E_sca": Esca, "epsilon": epsilon})
        plot_J_figure_multifreqs(currents, save_base + ".png", self.args.grid_num, self.num_freqs, self.args.N_inc)
        plot_params_figure(epsilon, save_base + "_params.png")
