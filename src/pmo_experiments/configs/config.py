"""Configuration management for pmo-experiments."""

import os
from typing import Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "model_configs", "default_config.yaml"
)

MODEL_ZOO_CFG_IDENTIFIER = "<pmo_experiments>::"

ConfigType = DictConfig


def _load_dict_config(config_path: str) -> DictConfig:
    """
    Load a configuration file as a DictConfig.

    Args:
        config_path (str): Path to the configuration YAML file.

    Raises:
        ValueError: If the loaded configuration is not a DictConfig.

    Returns:
        DictConfig: Loaded configuration object.

    """
    cfg = OmegaConf.load(config_path)

    if not isinstance(cfg, DictConfig):
        raise ValueError("Configuration file must be a DictConfig.")

    return cfg


def _merge_dict_configs(*configs: DictConfig) -> DictConfig:
    """
    Merge the configuration files into one.

    The output has to be a DictConfig.

    Args:
        *configs (DictConfig): Configuration objects to merge.

    Raises:
        ValueError: If the merged configuration is not a DictConfig.

    Returns:
        DictConfig: Merged configuration object.

    """
    merged_cfg = OmegaConf.merge(*configs)

    if not isinstance(merged_cfg, DictConfig):
        raise ValueError("Merged configuration must be a DictConfig.")

    return merged_cfg


def load_default_config() -> DictConfig:
    """
    Load the default configuration.

    Returns:
        DictConfig: Default configuration object.

    """
    return _load_dict_config(DEFAULT_CONFIG_PATH)


def merge_with_default(cfg: DictConfig) -> DictConfig:
    """
    Merge the config with the default config to fill missing values.

    Args:
        cfg (DictConfig): Configuration to merge with default.

    Returns:
        DictConfig: Merged configuration object. The original config values take
        precedence.

    """
    default_cfg = load_default_config()
    merged_cfg = _merge_dict_configs(default_cfg, cfg)

    return merged_cfg


def save_config(cfg: DictConfig, save_path: str) -> None:
    """
    Save a configuration object to a YAML file.

    Args:
        cfg (DictConfig): Configuration object to save.
        save_path (str): Path to the output YAML file.

    """
    OmegaConf.save(cfg, save_path)


def resolve_config_path(config_path: str, root_folder_name: str | None = None) -> str:
    """
    Resolve the name of a config_file.

    Automatically resolves model zoo configs and relative paths.

    Args:
        config_path (str): Name or path of the configuration file.
        root_folder_name (str | None, optional): Root folder for relative paths.
            Defaults to None.

    Raises:
        ValueError: If the file is not a yaml file.

    Returns:
        str: The resolved absolute path to the configuration file.

    """
    if not config_path.endswith(".yaml"):
        raise ValueError("Configuration file must be a .yaml file.")

    if config_path.startswith(MODEL_ZOO_CFG_IDENTIFIER):
        cfg_name = config_path[len(MODEL_ZOO_CFG_IDENTIFIER) :]
        model_zoo_base_folder = os.path.join(os.path.dirname(__file__), "model_configs")
        config_path = os.path.join(model_zoo_base_folder, cfg_name)
    elif root_folder_name is not None:
        config_path = os.path.join(root_folder_name, config_path)

    return os.path.abspath(config_path)


def get_config(config_path: str | None = None) -> DictConfig:
    """
    Load and merge configuration from a YAML file.

    Loads all base configurations recursively if specified.
    If MERGE_DEFAULT is set in the config, merges with the default config.

    Args:
        config_path (str | None): Path to the configuration YAML file. If None, load
            the default configuration. Defaults to None.

    Returns:
        DictConfig: Loaded and merged configuration object.

    """
    if config_path is None:
        return load_default_config()

    config_path = resolve_config_path(config_path)
    cfg = _load_dict_config(config_path)

    base_configs = []

    if cfg.get("__BASE__") is not None:
        if isinstance(cfg["__BASE__"], (list, tuple, ListConfig)):
            base_config_names = cfg["__BASE__"]
        else:
            base_config_names = [cfg["__BASE__"]]

        updated_base_config_names = []
        for base_config_name in base_config_names:
            resolved_base_config_path = resolve_config_path(
                base_config_name, root_folder_name=os.path.dirname(config_path)
            )

            assert os.path.abspath(resolved_base_config_path) != os.path.abspath(
                config_path
            ), "__BASE__ config cannot be the same as the main config."

            assert os.path.abspath(resolved_base_config_path) != os.path.abspath(
                DEFAULT_CONFIG_PATH
            ), (
                "__BASE__ config cannot be the same as the default config." + " Use "
                "MERGE_DEFAULT instead."
            )

            updated_base_config_names.append(resolved_base_config_path)
            base_configs.append(get_config(resolved_base_config_path))

        # Update __BASE__ to contain resolved paths
        cfg["__BASE__"] = updated_base_config_names

    cfg = _merge_dict_configs(*base_configs, cfg)
    current_merge_default = cfg.get("MERGE_DEFAULT")
    if current_merge_default is not None and current_merge_default:
        cfg = merge_with_default(cfg)

    return cfg


def config_from_command_line(cfg: DictConfig, overrides: Sequence[str]) -> DictConfig:
    """
    Update configuration based on command line overrides.

    Args:
        cfg (DictConfig): Config object to update.
        overrides (Sequence[str]): Configuration overrides in the format key=value.

    Returns:
        DictConfig: Updated configuration object.

    """
    if len(overrides) > 0:
        overrides_cfg = OmegaConf.from_dotlist(list(overrides))
        assert isinstance(overrides_cfg, DictConfig), "Overrides must be a DictConfig."

        updated_cfg = OmegaConf.merge(cfg, overrides_cfg)

        assert isinstance(updated_cfg, DictConfig), (
            "Updated configuration must be a DictConfig."
        )

        return updated_cfg

    return cfg
