from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


img2mse = lambda x, y: torch.mean((x - y) ** 2)


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._create_embedding_fn()

    def _create_embedding_fn(self) -> None:
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        n_freqs = self.kwargs["num_freqs"]
        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=n_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** max_freq, steps=n_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)


def get_embedder(multires: int, i_embed: int = 0):
    if i_embed == -1:
        return nn.Identity(), 2

    embed_kwargs = {
        "include_input": True,
        "input_dims": 2,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class NeJF(nn.Module):
    def __init__(
        self,
        D: int = 8,
        W: int = 256,
        input_ch: int = 2,
        output_ch: int = 1,
        skips=None,
        tanh: Optional[bool] = None,
        scale: float = 1e-5,
        epsilon: bool = False,
    ) -> None:
        super().__init__()
        if skips is None:
            skips = [4]
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.tanh = tanh
        self.epsilon = epsilon

        self.pts_linears = nn.ModuleList()
        in_dim = input_ch
        for i in range(D):
            if i - 1 in self.skips:
                in_dim += input_ch
            layers = [
                nn.Linear(in_dim, W),
                nn.BatchNorm1d(W),
                nn.GELU(),
            ]
            layers.append(nn.Dropout(0.1) if i % 2 == 0 else nn.Identity())
            self.pts_linears.append(nn.Sequential(*layers))
            in_dim = W

        self.output_linear = nn.Sequential(
            nn.Linear(W, W // 2),
            nn.BatchNorm1d(W // 2),
            nn.SiLU(),
            nn.Linear(W // 2, output_ch),
        )
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        input_pts, _ = torch.split(x, [self.input_ch, 0], dim=-1)
        h = x
        for i, block in enumerate(self.pts_linears):
            if i == 0:
                h = block(h)
                h_shortcut = h
            elif i % 3 == 0:
                h = (block(h) + h_shortcut) / 2.0
                h_shortcut = h
            else:
                h = block(h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        outputs = self.output_linear(h)
        if self.tanh is not None:
            if self.epsilon:
                outputs = 0.5 * (torch.tanh(outputs) + 1.0) + 1.0
            else:
                outputs = outputs * self.scale
        return outputs


__all__ = ["img2mse", "get_embedder", "CurrentNetwork", "NeJF"]
