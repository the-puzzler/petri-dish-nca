# Petri Dish NCA

This repo now contains a first-pass implementation of the `new.md` proposal:
competing NCAs on a fixed sine-derived environment tensor, using per-agent
encoder/decoder competition instead of attack/defense cosine similarity.

## Setup 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh         # Or install uv another way

git clone https://github.com/SakanaAI/petri-dish-nca
cd petri-dish-nca
uv sync
```

## Commands

- basic: `uv run python src/train.py --config configs/tiny-config.json`
- wandb logging: `uv run python src/train.py --config configs/example.json --wandb`
- run with config: `uv run python src/train.py --config configs/example.json`

## Configs

For additional configurations, you can load a JSON config file. Any parameters not specified in the config file will be set their default value in `src/config.py`
