import argparse
import os
import time
from src.utils.common import read_yaml, create_directories
from src.utils.callbacks import create_and_save_tensorboard_callback,create_and_save_checkpointing_callback

def prepare_callbacks(config_path: str) -> None:
    """prepare and save callbacks as binary
    Args:
        config_path (str): path to configuration file
    """
    config = read_yaml(config_path)
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    tensorboard_log_dir = os.path.join(artifacts_dir, artifacts["TENSORBOARD_ROOT_LOG_DIR"])

    checkpoint_dir = os.path.join(artifacts_dir, artifacts["CHECKPOINT_DIR"])

    callbacks_dir = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])

    create_directories([
        tensorboard_log_dir,
        checkpoint_dir,
        callbacks_dir
    ])

    create_and_save_tensorboard_callback(callbacks_dir, tensorboard_log_dir)
    create_and_save_checkpointing_callback(callbacks_dir, checkpoint_dir)





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        prepare_callbacks(config_path=parsed_args.config)
    except Exception as e:
        raise e