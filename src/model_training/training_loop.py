import time
import os
import requests
import logging
from typing import Optional

import tensorflow as tf
import numpy as np
from ray import train
from ray.train import Checkpoint

from .game_env import Game_Env
from .models.proximal_policy_optimization import Agent
from .Controller import Controller

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url, filename):
    logger.info(f"Downloading {url} to {filename}")
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    return filename


def training_loop(training_loop_config: dict):
    training_loop_config['env_config']['gb_path'] = download_file(training_loop_config['gb_path'], 'rom.gb')
    training_loop_config['checkpoint_dir'] = None

    checkpoint: Optional[Checkpoint] = train.get_checkpoint()
    logger.info(f"Checkpoint: {checkpoint}")
    if checkpoint:
        logger.info(f"Loading checkpoint: {checkpoint._uuid}")
        checkpoint_dir = checkpoint.as_directory()
        training_loop_config['env_config']['init_state'] = os.path.join(checkpoint_dir, 'init.state')
        training_loop_config['checkpoint_dir'] = checkpoint_dir
    else:
        logger.info(f"Downloading init.state: {training_loop_config['init_state']}")
        training_loop_config['env_config']['init_state'] = download_file(training_loop_config['init_state'], 'init.state')

    # Create Environment
    env = Game_Env(training_loop_config['env_config'], num_env=0)

    controller = Controller(env, training_loop_config)

    controller.train()
