from pathlib import Path

import yaml


def _cast(t, v: str):
    """Casts the command-line string `v` to the type of the config value `t` it overrides."""
    if v in ("null", "~"):
        return None
    if t is None:
        return yaml.safe_load(v)
    if isinstance(t, str):
        return v
    if isinstance(t, bool):
        value = yaml.safe_load(v)
        if not isinstance(value, bool):
            raise TypeError(f"Expected true/false, got `{v}`.")
        return value
    if isinstance(t, (int, float)):
        # int()/float() rather than YAML, which reads e.g. `1e-4` as a string.
        return type(t)(v)
    value = yaml.safe_load(v)
    if isinstance(t, list) and not isinstance(value, list):
        value = [value]
    if not isinstance(value, type(t)):
        raise TypeError(f"Expected a {type(t).__name__}, got `{v}`.")
    return value


def parse_config_overrides(config: dict, overrides: list[str]) -> dict:
    """Takes a list of args and generates an override dict from it.

    Args:
        config (dict): Original config dict to override. Used for type information.
        overrides (list[str]): Alternating `--key value` args.

    Raises:
        KeyError: If there is an arg present in the override list that is not found in the config file.
        ValueError: If an arg is malformed or its value can't be parsed.
        TypeError: If an arg's value doesn't match the type in the config file.

    Returns:
        dict: The override dictionary to use to use with `config.update()`.
    """
    result = {}
    args = list(overrides)

    while args:
        k = args.pop(0)
        if not k.startswith("--"):
            raise ValueError(f"Expected a `--key value` config override, got `{k}`.")
        k = k[2:]
        if not args or args[0].startswith("--"):
            raise ValueError(f"Config override `--{k}` is missing a value.")
        v = args.pop(0)

        if k not in config:
            raise KeyError(f"Config override `--{k}` is not a key in the config file.")

        try:
            result[k] = _cast(config[k], v)
        except (TypeError, ValueError) as e:
            raise type(e)(f"Invalid value for config override `--{k}`: {e}") from e

    return result


def _config_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _config_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(
        path: Path,
        print_loaded_configs: bool = False,
        _loaded: frozenset[Path] = frozenset()) -> dict:
    """Loads a config YAML.  Supports merging parent and child YAML with the __import__ key.
    Parent configs will always overwrite their children.

    Args:
        path (Path): Path to the YAML file.
        print_loaded_configs (bool, optional): Prints the full tree of merged configs. Defaults to False.
        _loaded (frozenset[Path], optional): Used for circular import detection, should not be used except for inside recursive calls.

    Raises:
        ValueError: In the event of a circular import.

    Returns:
        dict: The merged config dictionary.
    """
    path = Path(path).resolve()
    is_root = len(_loaded) == 0

    if print_loaded_configs and is_root:
        print("\nLoading configs from:")

    # Circular import detection
    if path in _loaded:
        raise ValueError(f"Circular config import detected:\n"
                         f"Import chain: {' -> '.join(str(p) for p in _loaded)} -> {path}")
    _loaded = _loaded | {path}

    # Load config
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    # Retrieve sub-configs to import
    import_cfgs = cfg.pop("__import__", [])
    if isinstance(import_cfgs, str):
        import_cfgs = [import_cfgs]

    # Merge all child configs in the import list.
    merged_cfg = {}
    for import_cfg_path in import_cfgs:
        child = load_config(path.parent / import_cfg_path, print_loaded_configs, _loaded)
        merged_cfg = _config_merge(merged_cfg, child)

    if print_loaded_configs:
        indent = "\t" * (len(_loaded) - 1)
        print(f"{indent}{'[ROOT]' if is_root else '<'} {path.name}")
        if is_root:
            print()

    # Parent overrides all children configs.
    merged = _config_merge(merged_cfg, cfg)
    
    if is_root:
        merged["__path__"] = str(path)
        
    return merged
