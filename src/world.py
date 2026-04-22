from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat

from config import Config
from model import CASunGroup


@contextmanager
def device_autocast(device, dtype=torch.bfloat16):
    if device == "cuda":
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield


class Feature:
    def __init__(self, config):
        pass

    def on_init(self, world):
        pass

    def before_step(self, world):
        pass

    def after_step(self, world, grid):
        pass


class SimpleBurnInFeature(Feature):
    def __init__(self, config: Config):
        self.initial_steps_before = 0
        self.initial_steps_per = 1
        self.target_steps_before = config.steps_before_update
        self.target_steps_per = config.steps_per_update
        self.increment = config.burn_in_increment
        self.increment_epochs = config.burn_in_increment_epochs

    def on_init(self, world):
        world.state = getattr(world, "state", {})
        world.state["steps_before_update"] = self.initial_steps_before
        world.state["steps_per_update"] = self.initial_steps_per
        world.state["last_burn_in_epoch"] = 0

    def before_step(self, world):
        if world.epoch - world.state["last_burn_in_epoch"] >= self.increment_epochs:
            world.state["steps_before_update"] = min(
                world.state["steps_before_update"] + self.increment,
                self.target_steps_before,
            )
            world.state["steps_per_update"] = min(
                world.state["steps_per_update"] + self.increment, self.target_steps_per
            )
            world.state["last_burn_in_epoch"] = world.epoch


class UpdatePoolWithNondeadFeature(Feature):
    def __init__(self, config: Config):
        self.n_ncas = config.n_ncas

    def after_step(self, world, grid):
        nca_growth = grid[:, 1 : self.n_ncas + 1].sum(dim=(2, 3))
        alive_mask = torch.all(nca_growth > 0, dim=1)
        world.pool[world.state["pool_idxs"][alive_mask]] = grid[alive_mask].detach()


class World:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = config.device
        self.epoch = 0
        self.steps_taken = 0
        self.batch_size = config.batch_size
        self.pool_size = config.pool_size
        self.seed_dim = config.cell_hidden_dim
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float

        self.state = {
            "steps_before_update": config.steps_before_update,
            "steps_per_update": config.steps_per_update,
            "pool_idxs": [],
            "mode": config.mode,
        }

        self._init_pool(config)
        self._init_environment(config)
        self.features = self._build_features(config)

        for feature in self.features:
            feature.on_init(self)

    def get_seed(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch_size < self.pool_size:
            self.state["pool_idxs"] = torch.randint(
                self.pool_size, size=(self.batch_size,), device=self.device
            )
        else:
            self.state["pool_idxs"] = torch.arange(self.batch_size, device=self.device)

        idxs = self.state["pool_idxs"]
        return self.pool[idxs], self.env_pool[idxs]

    def step(self, group: CASunGroup, grid: torch.Tensor, env: torch.Tensor):
        for feature in self.features:
            feature.before_step(self)

        steps_before = self.state.get("steps_before_update", 0)
        steps_per = self.state.get("steps_per_update", 1)
        epoch_steps = steps_before + steps_per
        grid_storage = torch.zeros(
            (epoch_steps, *grid.shape), device=self.device, dtype=grid.dtype
        )

        with device_autocast(self.device):
            with torch.no_grad():
                if steps_before:
                    grid, grids, _ = group(grid, env, steps_before)
                    grid_storage[:steps_before] = grids

        with device_autocast(self.device):
            grid, grids, forward_stats = group(grid, env, steps_per)
            grid_storage[steps_before:] = grids

        for feature in self.features:
            feature.after_step(self, grid)

        self.epoch += 1
        self.steps_taken += epoch_steps
        return self._get_stats_and_new_grid(group, grid, grid_storage.detach(), forward_stats)

    def save(self, config: Config, run_name: str) -> None:
        np.save(
            f"{run_name}/environment.npy",
            self.env_pool.float().detach().cpu().numpy(),
        )

    def _hidden_slice(self, agent_idx: int) -> slice:
        return self.config.agent_h_slice(agent_idx)

    def _init_pool(self, config: Config):
        self.pool = torch.zeros(
            config.pool_size,
            config.cell_dim,
            *config.grid_size,
            device=config.device,
            dtype=self.dtype,
        )

        all_coords = torch.cartesian_prod(
            torch.arange(config.grid_size[0], device=config.device),
            torch.arange(config.grid_size[1], device=config.device),
        )
        seed_idxs = torch.stack(
            [
                torch.randperm(config.total_grid_size, device=config.device)[
                    : config.n_ncas * config.n_seeds
                ].view(config.n_ncas, config.n_seeds)
                for _ in range(config.pool_size)
            ]
        )
        seed_pts = all_coords[seed_idxs]
        init_xs = seed_pts[:, :, :, 0]
        init_ys = seed_pts[:, :, :, 1]

        self.pool[:, 0] = 1.0
        for agent_idx in range(config.n_ncas):
            if config.seed_mode == "solid":
                seed_vals = torch.ones(
                    (config.pool_size, config.n_seeds, self.seed_dim),
                    device=config.device,
                    dtype=self.dtype,
                )
            else:
                seed_base = torch.randn(
                    (config.pool_size, self.seed_dim),
                    device=config.device,
                    dtype=self.dtype,
                )
                seed_base = F.normalize(seed_base, dim=-1)
                seed_vals = seed_base.unsqueeze(1).repeat(1, config.n_seeds, 1)

            batch_ids = (
                torch.arange(config.pool_size, device=config.device)
                .unsqueeze(1)
                .expand(-1, config.n_seeds)
                .reshape(-1)
            )
            xs = init_xs[:, agent_idx].reshape(-1)
            ys = init_ys[:, agent_idx].reshape(-1)
            self.pool[batch_ids, agent_idx + 1, xs, ys] = 1.0

            self.pool[batch_ids, self.config.agent_a_slice(agent_idx), xs, ys] = 0.0
            self.pool[batch_ids, self.config.agent_d_slice(agent_idx), xs, ys] = 0.0
            hidden_slice = self._hidden_slice(agent_idx)
            self.pool[batch_ids, hidden_slice, xs, ys] = seed_vals.reshape(
                -1, self.seed_dim
            )

    def _encode_scalar_pairs(
        self, x_values: torch.Tensor, y_values: torch.Tensor
    ) -> torch.Tensor:
        env = torch.stack(
            [
                torch.cos(x_values),
                torch.sin(x_values),
                torch.cos(y_values),
                torch.sin(y_values),
            ],
            dim=-1,
        )
        return F.normalize(env, dim=-1)

    def _build_sample_table(self, config: Config) -> torch.Tensor:
        if config.env_kind == "sine":
            width = config.grid_size[1]
            x = torch.linspace(
                config.sine_x_min,
                config.sine_x_max,
                width,
                device=config.device,
                dtype=self.dtype,
            )
            y = torch.sin(x)
            return self._encode_scalar_pairs(x, y)

        if config.data_env_vectors is not None:
            samples = torch.tensor(
                config.data_env_vectors, device=config.device, dtype=self.dtype
            )
            if samples.shape[-1] != config.env_dim:
                raise ValueError("data_env_vectors width must equal env_dim")
            return F.normalize(samples, dim=-1)

        x_values = torch.tensor(config.data_x_values, device=config.device, dtype=self.dtype)
        y_values = torch.tensor(config.data_y_values, device=config.device, dtype=self.dtype)
        if x_values.shape != y_values.shape:
            raise ValueError("data_x_values and data_y_values must have same length")
        return self._encode_scalar_pairs(x_values, y_values)

    def _assign_samples_to_grid(self, samples: torch.Tensor, config: Config) -> torch.Tensor:
        height, width = config.grid_size
        total_cells = height * width
        num_samples = samples.shape[0]

        if config.data_assignment == "x_axis":
            column_ids = torch.linspace(
                0,
                num_samples - 1,
                width,
                device=config.device,
            ).round().long()
            env = samples[column_ids]
            env = repeat(env, "w c -> h w c", h=height)
        elif config.data_assignment == "random":
            sample_ids = torch.randint(
                num_samples, (total_cells,), device=config.device
            )
            env = samples[sample_ids].view(height, width, config.env_dim)
        else:
            sample_ids = torch.arange(total_cells, device=config.device) % num_samples
            env = samples[sample_ids].view(height, width, config.env_dim)

        return env.permute(2, 0, 1).contiguous()

    def _init_environment(self, config: Config) -> None:
        samples = self._build_sample_table(config)
        env = self._assign_samples_to_grid(samples, config)
        self.env_pool = repeat(env, "c h w -> b c h w", b=config.pool_size).contiguous()

    def _build_features(self, config: Config) -> list[Feature]:
        features: list[Feature] = []
        if config.burn_in:
            features.append(SimpleBurnInFeature(config))
        features.append(UpdatePoolWithNondeadFeature(config))
        return features

    def _get_stats_and_new_grid(
        self,
        group: CASunGroup,
        grid: torch.Tensor,
        grid_storage: torch.Tensor,
        forward_stats,
    ):
        stats = group.update_models(grid, forward_stats)
        return stats, grid.detach(), grid_storage
