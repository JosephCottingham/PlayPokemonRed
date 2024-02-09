import time
import os
import requests
import logging
from typing import Optional, List, Tuple, Dict, Any

import faulthandler
faulthandler.enable()

import tensorflow as tf
import numpy as np
import ray
from ray import train
from ray.train import Checkpoint, Result


from .game_env import Game_Env
from .models.proximal_policy_optimization import Agent
from .Controller import Controller

logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger("ray.train")

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
    if not checkpoint:
        logger.info(f"No checkpoint found for Session")

        # checkpoint: Optional[Checkpoint] = result.get_best_checkpoint(metric="avg_policy_network_loss", mode="min")
        if checkpoint:
            logger.info(f"Best checkpoint: {checkpoint}")
        else:
            logger.info(f"No best checkpoint found")
        
    logger.info(f"Checkpoint: {checkpoint}")
    
    # Load Checkpoint data and set directory, this includes loading the init.state file from the checkpoint.
    if checkpoint:
        checkpoint_dir = checkpoint.to_directory()
        logger.info(f"Loading checkpoint: {checkpoint._uuid}")
        logger.info(f"Checkpoint Directory: {checkpoint_dir}")
        my_list = os.listdir(checkpoint_dir)
        logger.info(f"Files in Checkpoint Directory: {my_list}")

        # Load init.state from checkpoint
        training_loop_config['env_config']['init_state'] = os.path.join(str(checkpoint_dir), 'init.state')
        training_loop_config['checkpoint_dir'] = str(checkpoint_dir)
        logger.info(f'checkpoint_dir: {training_loop_config["checkpoint_dir"]}')
    else:
        logger.info(f"Downloading init.state: {training_loop_config['init_state']}")
        training_loop_config['env_config']['init_state'] = download_file(training_loop_config['init_state'], 'init.state')
    
    # Collect Ray process IDs
    process_ids = {
        'actor_id': ray.get_runtime_context().get_actor_id(),
        'job_id': ray.get_runtime_context().get_job_id(),
        'worker_id': ray.get_runtime_context().get_worker_id()
    }
    logger.info(f'Actor ID: {process_ids["actor_id"]}, Job ID: {process_ids["job_id"]}, Worker ID: {process_ids["worker_id"]}')
    training_loop_config['screenshots_s3_path'] += process_ids["actor_id"]


    # Create Environment
    env = Game_Env(training_loop_config['env_config'], num_env=0)

    controller = Controller(env, training_loop_config, checkpoint, process_ids)

    controller.train()
