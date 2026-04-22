import json
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class Config:
    # Grid
    grid_size: tuple[int, int] = (64, 64)
    n_seeds: int = 1

    # World state
    cell_hidden_dim: int = 8
    alive_visible: bool = True
    alive_threshold: float = 0.4

    # Fixed environment
    env_kind: Literal["sine", "table"] = "sine"
    env_dim: int = 4
    sine_x_min: float = -3.141592653589793
    sine_x_max: float = 3.141592653589793
    data_assignment: Literal["x_axis", "tile", "random"] = "x_axis"
    data_x_values: list[float] | None = None
    data_y_values: list[float] | None = None
    data_env_vectors: list[list[float]] | None = None

    # Seeding
    seed_dist: Literal["scatter"] = "scatter"
    seed_mode: Literal["solid", "random"] = "random"

    # Burn-in config
    burn_in: bool = False
    burn_in_increment_epochs: int = 0
    burn_in_increment: int = 0

    # NCAs
    n_ncas: int = 6
    n_hidden_layers: int = 1
    hidden_dim: int = 32
    latent_dim: int = 4
    decoder_hidden_dim: int = 16
    model_kernel_size: int = 3
    model_dropout_per: float = 0.0

    # Training
    softmax_temp: float = 1.0
    reconstruction_loss_scale: float = 1.0
    optimizer: Literal["AdamW", "Adam", "RMSProp", "SGD"] = "Adam"
    learning_rate: float = 3e-4
    batch_size: int = 8
    pool_size: int = 64
    epochs: int = 1_000
    log_every: int = 100
    wandb: bool = False

    # Multi-world
    steps_before_update: int = 0
    steps_per_update: int = 4
    region_bins: int = 8

    # General system
    device: Literal["cpu", "cuda", "mps"] = "cuda"
    seed: int = 42
    mode: Literal["train", "eval", "frozen_eval"] = "train"

    def __post_init__(self) -> None:
        assert self.batch_size <= self.pool_size, "[config] batch_size > pool_size"
        assert self.n_seeds * self.n_ncas <= self.total_grid_size, (
            "[config] n_seeds * n_ncas > self.total_grid_size"
        )
        assert self.softmax_temp > 0, "[config] softmax_temp <= 0"
        assert self.region_bins > 0, "[config] region_bins must be positive"
        if self.env_kind == "sine":
            assert self.env_dim == 4, "[config] sine experiment expects env_dim == 4"
        if self.env_kind == "table":
            has_vectors = self.data_env_vectors is not None
            has_xy = self.data_x_values is not None and self.data_y_values is not None
            assert has_vectors or has_xy, (
                "[config] table environment needs data_env_vectors or data_x_values/data_y_values"
            )

        if self.device == "cuda" and not torch.cuda.is_available():
            print("[warning] CUDA not available, falling back to CPU")
            object.__setattr__(self, "device", "cpu")
        elif self.device == "mps" and not torch.backends.mps.is_available():
            print("[warning] MPS not available, falling back to CPU")
            object.__setattr__(self, "device", "cpu")

        self._set_random_seed()

    def _set_random_seed(self) -> None:
        import random

        import numpy as np

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @property
    def alive_dim(self) -> int:
        return self.n_ncas + 1

    @property
    def agent_state_dim(self) -> int:
        return self.latent_dim + self.env_dim + self.cell_hidden_dim

    @property
    def cell_dim(self) -> int:
        return self.alive_dim + self.n_ncas * self.agent_state_dim

    def agent_state_offset(self, agent_idx: int) -> int:
        return self.alive_dim + agent_idx * self.agent_state_dim

    def agent_a_slice(self, agent_idx: int) -> slice:
        start = self.agent_state_offset(agent_idx)
        end = start + self.latent_dim
        return slice(start, end)

    def agent_d_slice(self, agent_idx: int) -> slice:
        start = self.agent_state_offset(agent_idx) + self.latent_dim
        end = start + self.env_dim
        return slice(start, end)

    def agent_h_slice(self, agent_idx: int) -> slice:
        start = self.agent_state_offset(agent_idx) + self.latent_dim + self.env_dim
        end = start + self.cell_hidden_dim
        return slice(start, end)

    @property
    def total_grid_size(self) -> int:
        return self.grid_size[0] * self.grid_size[1]

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(**json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
