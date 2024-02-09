import time
import os
import requests
import logging
from typing import Optional
import tempfile
import subprocess

import faulthandler
faulthandler.enable()

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

def download_s3_dir(s3_uri, local_dir):
    """
    params:
    - s3_uri: S3 URI of the directory to download
    - local_dir: Local path to download the directory to
    """
    subprocess.check_call(['aws', 's3', 'cp', s3_uri, local_dir, '--recursive'])

def play_loop(training_loop_config: dict):

    training_loop_config['env_config']['gb_path'] = download_file(training_loop_config['gb_path'], 'rom.gb')
    
    if training_loop_config.get('checkpoint_dir'):
        # Download Checkpoint
        checkpointDir = tempfile.mkdtemp()
        download_s3_dir(training_loop_config['checkpoint_dir'], checkpointDir)
        training_loop_config['checkpoint_dir'] = checkpointDir
        training_loop_config['env_config']['init_state'] = os.path.join(checkpointDir, 'init.state')
    else:
        training_loop_config['env_config']['init_state'] = download_file(training_loop_config['init_state'], 'init.state')

    # Create Environment
    env = Game_Env(training_loop_config['env_config'], num_env=0)

    controller = Controller(env, training_loop_config)

    controller.play()
