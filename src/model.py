import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class AgentAutoencoder(nn.Module):
    """Per-agent autoencoder with neighborhood-aware hidden-state dynamics."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        padding = (config.model_kernel_size - 1) // 2
        self.context = nn.Sequential(
            nn.Conv2d(
                config.env_dim + config.cell_hidden_dim,
                config.hidden_dim,
                config.model_kernel_size,
                padding=padding,
            ),
            nn.GELU(),
            nn.Dropout(p=config.model_dropout_per),
        )

        reasoning_layers = []
        for _ in range(config.n_hidden_layers):
            reasoning_layers.extend(
                [
                    nn.Conv2d(config.hidden_dim, config.hidden_dim, 1),
                    nn.GELU(),
                    nn.Dropout(p=config.model_dropout_per),
                ]
            )
        self.reasoning = nn.Sequential(*reasoning_layers)

        self.encoder = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, config.latent_dim, 1),
            nn.Tanh(),
        )
        self.hidden_update = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, config.cell_hidden_dim, 1),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(config.latent_dim, config.decoder_hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(config.decoder_hidden_dim, config.env_dim, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        env: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.context(torch.cat([env, hidden], dim=1))
        if len(self.reasoning) > 0:
            features = self.reasoning(features)

        latent = F.normalize(self.encoder(features), dim=1, eps=1e-6)
        reconstruction = self.decoder(latent).clamp(-1, 1)
        hidden_update = self.hidden_update(features)
        return latent, reconstruction, hidden_update

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent).clamp(-1, 1)


class CASunGroup:
    """Competitive autoencoder NCAs on a fixed environment substrate."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.n_ncas = config.n_ncas
        self.N = self.n_ncas + 1
        self.device = config.device
        self.total_grid_size = config.total_grid_size
        self.softmax_temp = config.softmax_temp
        self.threshold = torch.tensor(config.alive_threshold, device=config.device)

        self.ali_idxs = torch.arange(self.N, device=config.device)
        self.agents = nn.ModuleList(
            [AgentAutoencoder(config) for _ in range(config.n_ncas)]
        ).to(config.device)
        self.optimizer = self._make_optimizer(config)
        self.last_metrics: dict[str, Any] | None = None

    def _make_optimizer(self, config: Config) -> torch.optim.Optimizer:
        optimizer_map = {
            "AdamW": torch.optim.AdamW,
            "Adam": torch.optim.Adam,
            "RMSProp": torch.optim.RMSprop,
            "SGD": torch.optim.SGD,
        }
        optimizer_class = optimizer_map.get(config.optimizer)
        return optimizer_class(self.agents.parameters(), lr=config.learning_rate)

    def _get_alive_mask(self, grid: torch.Tensor) -> torch.Tensor:
        alive_channels = grid[:, self.ali_idxs]
        alive_flat = alive_channels.reshape(-1, 1, *alive_channels.shape[-2:])
        alive_pooled = F.max_pool2d(alive_flat, 3, stride=1, padding=1)
        alive_mask = alive_pooled.view(
            grid.shape[0], self.N, *grid.shape[-2:]
        ) > self.threshold
        alive_mask[:, 0] = True
        return alive_mask

    def _get_agent_a(self, grid: torch.Tensor, agent_idx: int) -> torch.Tensor:
        return grid[:, self.config.agent_a_slice(agent_idx)]

    def _get_agent_d(self, grid: torch.Tensor, agent_idx: int) -> torch.Tensor:
        return grid[:, self.config.agent_d_slice(agent_idx)]

    def _get_agent_h(self, grid: torch.Tensor, agent_idx: int) -> torch.Tensor:
        return grid[:, self.config.agent_h_slice(agent_idx)]

    def _compute_scores(
        self,
        latents: list[torch.Tensor],
        env: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = env.shape
        cross_errors = torch.zeros(
            batch_size,
            self.n_ncas,
            self.n_ncas,
            height,
            width,
            device=self.device,
            dtype=env.dtype,
        )

        for decoder_idx, agent in enumerate(self.agents):
            for code_idx, latent in enumerate(latents):
                cross_errors[:, decoder_idx, code_idx] = (
                    (agent.decode(latent) - env) ** 2
                ).mean(dim=1)

        own_errors = torch.stack(
            [cross_errors[:, idx, idx] for idx in range(self.n_ncas)], dim=1
        )
        strengths = torch.zeros(
            batch_size, self.N, height, width, device=self.device, dtype=env.dtype
        )
        strengths[:, 1:] = -own_errors

        for i in range(self.n_ncas):
            for j in range(self.n_ncas):
                if i == j:
                    continue
                strengths[:, i + 1] += -cross_errors[:, i, j] + cross_errors[:, j, i]

        return strengths, own_errors, cross_errors

    def _forward_step(
        self,
        grid: torch.Tensor,
        env: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        alive_mask = self._get_alive_mask(grid)

        latents: list[torch.Tensor] = []
        reconstructions: list[torch.Tensor] = []
        hidden_updates: list[torch.Tensor] = []

        for agent_idx, agent in enumerate(self.agents):
            hidden = self._get_agent_h(grid, agent_idx)
            latent, reconstruction, hidden_update = agent(hidden, env)
            agent_alive = alive_mask[:, agent_idx + 1].unsqueeze(1)
            latents.append(latent * agent_alive)
            reconstructions.append(reconstruction)
            hidden_updates.append(hidden_update * agent_alive)

        strengths, own_errors, cross_errors = self._compute_scores(latents, env)
        strengths = strengths.masked_fill(~alive_mask, float("-inf"))
        weights = torch.softmax(strengths / self.softmax_temp, dim=1)

        new_grid = grid.clone()
        new_grid[:, self.ali_idxs] = weights

        for agent_idx in range(self.n_ncas):
            hidden = self._get_agent_h(grid, agent_idx)
            weighted_update = hidden_updates[agent_idx] * weights[:, agent_idx + 1].unsqueeze(1)
            new_grid[:, self.config.agent_a_slice(agent_idx)] = latents[agent_idx]
            new_grid[:, self.config.agent_d_slice(agent_idx)] = reconstructions[agent_idx]
            new_grid[:, self.config.agent_h_slice(agent_idx)] = (
                hidden + weighted_update
            ).clamp(-1, 1)

        next_alive_mask = self._get_alive_mask(new_grid)
        new_grid[:, self.ali_idxs] = new_grid[:, self.ali_idxs] * next_alive_mask
        alive_sum = new_grid[:, self.ali_idxs].sum(dim=1, keepdim=True).clamp_min(1e-6)
        new_grid[:, self.ali_idxs] = new_grid[:, self.ali_idxs] / alive_sum

        metrics = {
            "strengths": strengths,
            "own_errors": own_errors,
            "cross_errors": cross_errors,
        }
        return new_grid, metrics

    def __call__(
        self,
        grid: torch.Tensor,
        env: torch.Tensor,
        steps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
        all_grids = torch.zeros(
            (steps, *grid.shape), device=grid.device, dtype=grid.dtype
        )
        step_metrics = None

        for step_idx in range(steps):
            grid, step_metrics = self._forward_step(grid, env)
            all_grids[step_idx].copy_(grid.detach())

        self.last_metrics = (
            {key: value.detach() for key, value in step_metrics.items()}
            if step_metrics is not None
            else None
        )
        return grid, all_grids, step_metrics

    def _region_means(self, tensor: torch.Tensor) -> torch.Tensor:
        width = tensor.shape[-1]
        bins = min(self.config.region_bins, width)
        edges = torch.linspace(0, width, bins + 1, device=tensor.device).round().long()
        regions = []
        for idx in range(bins):
            start = edges[idx].item()
            end = max(edges[idx + 1].item(), start + 1)
            regions.append(tensor[..., start:end].mean(dim=-1))
        return torch.stack(regions, dim=-1)

    def update_models(
        self,
        grid: torch.Tensor,
        metrics: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        if metrics is None:
            raise ValueError("Expected step metrics for update")

        alive_channels = grid[:, self.ali_idxs]
        batch_alive = alive_channels.view(grid.shape[0], self.N, -1).sum(-1).mean(0)
        log_growth = torch.asinh(batch_alive + 1e-3)

        own_errors = metrics["own_errors"]
        cross_errors = metrics["cross_errors"]
        mean_mse = own_errors.mean(dim=(0, 2, 3))
        loss = (-log_growth[1:] + self.config.reconstruction_loss_scale * mean_mse).sum()

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agents.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        percent_covered = batch_alive / self.total_grid_size * 100.0
        cross_matrix = cross_errors.mean(dim=(0, 3, 4))
        territory_profile = alive_channels[:, 1:].mean(dim=(0, 2))
        mse_profile = own_errors.mean(dim=(0, 2))
        territory_regions = self._region_means(territory_profile)
        mse_regions = self._region_means(mse_profile)
        region_winners = territory_regions.argmax(dim=0)

        return {
            "loss": loss.detach().item(),
            "growth": percent_covered.tolist(),
            "grad_norm": float(grad_norm),
            "reconstruction_mse": mean_mse.detach().cpu().tolist(),
            "cross_mse_matrix": cross_matrix.detach().cpu().tolist(),
            "territory_by_region": territory_regions.detach().cpu().tolist(),
            "mse_by_region": mse_regions.detach().cpu().tolist(),
            "region_winners": region_winners.detach().cpu().tolist(),
        }

    def save(self, config: Config, run_name: str) -> None:
        os.makedirs(run_name, exist_ok=True)
        config.save(f"{run_name}/config.json")
        torch.save(
            {
                "optimizer_state_dict": self.optimizer.state_dict(),
                "agents_state_dict": self.agents.state_dict(),
            },
            f"{run_name}/model.pt",
        )

    def load(self, loc: str) -> bool:
        checkpoint = torch.load(f"{loc}/model.pt", map_location=self.device)
        self.agents.load_state_dict(checkpoint["agents_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return True
