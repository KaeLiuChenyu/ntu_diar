import os
import yaml
import logging
from .data_utils import recursive_update


def setup_logging(
    config_path="log-config.yaml", overrides={}, default_level=logging.INFO,
):
    if os.path.exists(config_path):
        with open(config_path, "rt") as f:
            config = yaml.safe_load(f)
        recursive_update(config, overrides)
        logging.config.dictConfig(config)
    else:
        print("basic")
        logging.basicConfig(level=default_level)