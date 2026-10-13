"""Normalize explicit task-config CLI overrides without importing the VLM."""

from __future__ import annotations

from omegaconf import OmegaConf


def load_config(path: str, overrides: list[str]):
    """Load YAML, accepting both ``--section.key value`` and ``section.key=value``.

    Later arguments take precedence. YAML handles numbers, booleans and null;
    shell quoting preserves paths with spaces. Misspelled keys fail explicitly.
    """
    cfg = OmegaConf.load(path)
    dotlist = []
    index = 0
    while index < len(overrides):
        argument = overrides[index]
        index += 1
        if "=" in argument:
            key, value = argument.removeprefix("--").split("=", 1)
        elif argument.startswith("--"):
            key = argument[2:]
            if index == len(overrides) or overrides[index].startswith("--"):
                raise ValueError(f"Missing value for {argument}; use '{argument} VALUE'")
            value = overrides[index]
            index += 1
        else:
            raise ValueError(f"Invalid override {argument!r}; use --section.key VALUE or section.key=VALUE")
        missing = object()
        if OmegaConf.select(cfg, key, default=missing) is missing:
            raise ValueError(f"Unknown configuration key: {key}")
        dotlist.append(f"{key}={value}")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    if cfg.framework.name != "QwenFast" or cfg.framework.action_model.action_dim != 7:
        raise ValueError("This task requires QwenFast with seven absolute action targets")
    return cfg
