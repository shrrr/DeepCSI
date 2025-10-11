import csv
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import torch
import torch.nn as nn

try:
    import mindspore
    from mindspore import Tensor as msTensor
    from mindspore import ops as ms_ops
except ImportError:
    mindspore = None
    msTensor = None
    ms_ops = None


class TotalVariationL1(nn.Module):
    """Anisotropic total-variation regulariser with L1 norm."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.dim() != 2:
            raise ValueError("TV loss expects a 2D permittivity map")
        dx = torch.abs(x[1:, :] - x[:-1, :]).mean()
        dy = torch.abs(x[:, 1:] - x[:, :-1]).mean()
        return self.weight * (dx + dy)


class TotalVariationL2(nn.Module):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.dim() != 2:
            raise ValueError("TV loss expects a 2D permittivity map")
        dx = torch.pow(x[1:, :] - x[:-1, :], 2).mean()
        dy = torch.pow(x[:, 1:] - x[:, :-1], 2).mean()
        return self.weight * (dx + dy)


class MultiDirectionalTV(nn.Module):
    def __init__(self, main_directions=None, weight: float = 1.0, gamma: float = 0.7) -> None:
        super().__init__()
        if main_directions is None:
            main_directions = [(0, 1), (1, 0)]
        self.main_directions = main_directions
        self.other_directions = [(1, 1), (1, -1), (0, -1), (-1, 0), (-1, 1), (-1, -1)]
        self.weight = weight
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        B, C, H, W = x.shape
        total = 0.0
        for dy, dx in self.main_directions:
            h_start = max(-dy, 0)
            h_end = H - max(dy, 0)
            w_start = max(-dx, 0)
            w_end = W - max(dx, 0)
            src = x[:, :, h_start:h_end, w_start:w_end]
            tgt = x[:, :, h_start + dy : h_end + dy, w_start + dx : w_end + dx]
            total += self.gamma * torch.abs(src - tgt).mean()
        for dy, dx in self.other_directions:
            h_start = max(-dy, 0)
            h_end = H - max(dy, 0)
            w_start = max(-dx, 0)
            w_end = W - max(dx, 0)
            src = x[:, :, h_start:h_end, w_start:w_end]
            tgt = x[:, :, h_start + dy : h_end + dy, w_start + dx : w_end + dx]
            total += torch.abs(src - tgt).mean()
        return self.weight * total


def epsilon_control(field: torch.Tensor, use_constraint: bool = False) -> torch.Tensor:
    if use_constraint:
        field = torch.tanh(field)
        field = 0.5 * (field + 1.0) + 1.0
    return field


def _mindspore_reduce_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    if mindspore is None or msTensor is None or ms_ops is None:
        return float(values.mean().item())
    ms_tensor = msTensor(values.detach().cpu().numpy(), dtype=mindspore.float32)
    reduce_mean = ms_ops.ReduceMean(keep_dims=False)
    return float(reduce_mean(ms_tensor).asnumpy().item())


@dataclass
class LossRecorder:
    save_dir: str
    loss_names: Iterable[str]
    history: Dict[str, List[float]] = field(init=False)
    _mindspore_means: Dict[str, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.save_dir = os.path.abspath(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.loss_names = list(self.loss_names)
        self.history = {name: [] for name in self.loss_names}
        self._mindspore_means = {name: 0.0 for name in self.loss_names}

    def update(self, values: Dict[str, float]) -> None:
        for name in self.loss_names:
            if name in values:
                self.history[name].append(values[name])
                if mindspore is not None and msTensor is not None and ms_ops is not None:
                    torch_values = torch.tensor(self.history[name], dtype=torch.float32)
                    self._mindspore_means[name] = _mindspore_reduce_mean(torch_values)

    def plot_losses(self) -> None:
        if not self.history or plt is None:
            return
        plt.figure(figsize=(6, 4))
        for name, values in self.history.items():
            if values:
                plt.plot(values, label=name)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend(frameon=False)
        plt.tight_layout()
        path = os.path.join(self.save_dir, "loss_history.png")
        plt.savefig(path, dpi=200)
        plt.close()

    def save_history(self, filename: str = "loss_history.npy") -> None:
        path = os.path.join(self.save_dir, filename)
        np.save(path, self.history)

    def mindspore_mean(self, name: str) -> float:
        return self._mindspore_means.get(name, 0.0)


def save_results_row(filepath: str, header: Iterable[str], row: Iterable) -> None:
    filepath = os.path.abspath(filepath)
    ensure_parent_dir(filepath)
    need_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
    with open(filepath, "a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if need_header:
            writer.writerow(list(header))
        writer.writerow(list(row))


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def besselh(v: int, kind: int, z: torch.Tensor) -> torch.Tensor:
    if v not in (0, 1):
        raise NotImplementedError("Only orders 0 and 1 are supported")
    if kind not in (1, 2):
        raise NotImplementedError("Only Hankel functions of the first and second kind are supported")
    import torch.special as sp

    if v == 0:
        jv = sp.bessel_j0(z)
        yv = sp.bessel_y0(z)
    else:
        jv = sp.bessel_j1(z)
        yv = sp.bessel_y1(z)
    return jv + (1j if kind == 1 else -1j) * yv


def read_exp_data(filename: str, nf: int, ns: int, nr: int) -> torch.Tensor:
    data = np.loadtxt(filename, skiprows=10)
    p_sca = (data[:, 3] - data[:, 5]) + 1j * (data[:, 4] - data[:, 6])
    try:
        from scipy.io import loadmat  # type: ignore

        calib_file = os.path.join(os.path.dirname(filename), f"calib_rats_{os.path.basename(filename)[4:-6]}.mat")
        calib = loadmat(calib_file)["calib_rats"].ravel()
        for idx in range(data.shape[0]):
            s_idx = int(data[idx, 0]) - 1
            f_idx = int(data[idx, 2]) - 1
            coef = calib[s_idx * nf + f_idx - 1]
            p_sca[idx] /= coef
    except Exception:
        pass

    p_sca = p_sca.reshape(ns, nr, nf).transpose(2, 0, 1)
    return torch.tensor(p_sca, dtype=torch.complex64)


def data_in_need_from_vie(nf: int, nr: int, receiv_n: int, trans_n: int, device: torch.device) -> torch.Tensor:
    masks = torch.zeros((trans_n, receiv_n), device=device)
    for jj in range(trans_n):
        theta0 = (jj * 360) / trans_n
        theta_no_response = []
        for t in range(-59, 60):
            angle = int((theta0 + t) % receiv_n)
            theta_no_response.append(angle)
        mask = torch.ones(receiv_n, device=device)
        mask[theta_no_response] = 0
        masks[jj, :] = mask
    return (masks != 0).float()


def fresnel_data_preprocess(
    nf: int,
    nr: int,
    receiv_n: int,
    trans_n: int,
    device: torch.device,
) -> torch.Tensor:
    masks = data_in_need_from_vie(nf, nr, receiv_n, trans_n, device=device)
    masks_full = torch.zeros((trans_n, nr, receiv_n), device=device)
    for ii in range(trans_n):
        mask_ii = torch.zeros(nr, receiv_n, device=device)
        idx = torch.nonzero(masks[ii, :], as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        for hh in range(nr):
            mask_ii[hh, idx[hh % idx.numel()]] = 1
        masks_full[ii] = mask_ii
    return masks_full.to(torch.complex64)


def plot_J_figure_multifreqs(data: np.ndarray, filepath: str, grid_num: int, num_freqs: int, num_inc: int) -> None:
    if not data.shape[-1] == 2:
        data = np.concatenate([np.real(data)[...,None], np.imag(data)[...,None]], axis=-1)
    reshaped = data.reshape(num_freqs, grid_num, grid_num, num_inc, 2)    
    fig, axes = plt.subplots(num_freqs, num_inc, figsize=(25,6))
    axes = axes.reshape((num_freqs, num_inc))
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    for f_idx in range(num_freqs):
        for inc_idx in range(num_inc):
            ax = axes[f_idx, inc_idx]
            amp = np.linalg.norm(reshaped[f_idx, :, :, inc_idx], axis=-1)
            if 'fresnel' in filepath:
                im = ax.imshow((amp).T, cmap="viridis", origin="upper")
            else:
                im = ax.imshow((amp), cmap="viridis", origin="upper")
            ax.axis("off")
    fig.colorbar(im, cax=cbar_ax)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()

def plot_params_figure(epsilon: np.ndarray, filepath: str) -> None:
    if plt is None:
        return
    if epsilon.shape[1] == 1:
        epsilon = np.reshape(epsilon, (int(np.sqrt(epsilon.shape[0])), -1))
    plt.figure(figsize=(8, 6))
    if 'fresnel' in filepath:
        plt.imshow((epsilon), cmap="viridis", origin="upper", aspect='auto')
    else:
        plt.imshow((epsilon).T, cmap="viridis", origin="upper", aspect='auto')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()
