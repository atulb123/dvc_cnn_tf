import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories, copy_files
from src.utils.models import get_vgg_16_model, prepare_full_model
import random

STAGE = "STAGE_NAME"  ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)


def prepare_base_model(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    artifacts = config["artifacts"]
    ARTIFACTS_DIR = artifacts["ARTIFACTS_DIR"]
    BASE_MODEL_DIR = artifacts["BASE_MODEL_DIR"]
    BASE_MODEL_NAME = artifacts["BASE_MODEL_NAME"]
    BASE_MODEL_DIR_PATH = os.path.join(ARTIFACTS_DIR, BASE_MODEL_DIR)
    create_directories([BASE_MODEL_DIR_PATH])
    base_model_path = os.path.join(BASE_MODEL_DIR_PATH, BASE_MODEL_NAME)
    base_model = get_vgg_16_model(params["IMAGE_SIZE"], base_model_path)
    full_model = prepare_full_model(base_model=base_model, CLASSES=2, freeze_all=True, freeze_till=None,
                                    learning_rate=params["LEARNING_RATE"])
    updated_base_model_path = os.path.join(BASE_MODEL_DIR_PATH, artifacts["UPDATED_BASE_MODEL_NAME"])
    full_model.save(updated_base_model_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        prepare_base_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
