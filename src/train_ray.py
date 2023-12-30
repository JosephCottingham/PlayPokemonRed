import logging

import ray
from ray.runtime_env import RuntimeEnv
from ray.train.tensorflow import TensorflowTrainer
from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, FailureConfig

from model_training.training_loop import training_loop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

storage_path = '/home/joe/Projects/pokemon_checkpoints'
# storage_path = 's3://pokemon-ml/checkpoints/'

address='ray://192.168.1.239:10001'

run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        # *Best* checkpoints are determined by these params:
        checkpoint_score_attribute="avg_policy_network_loss",
        checkpoint_score_order="min",
    ),
    # This will store checkpoints on S3.
    storage_path=storage_path,
    failure_config=FailureConfig(
        max_failures=0,
    ),
)

training_loop_config = {
    'env_config': {
        'save_screenshots': False,
        'save_new_state_screen_shot': False,
        'tick_freq': 60,
        'emulation_speed': 5,
        'window_type': 'SDL2', #SDL2
        'disable_input': True,
    },
    'screen_shot_freq': 10,
    'model_save_path': 'C:\\Users\\josep\\Projects\\PlayPokemonRed\\artifacts',
    'max_episodes': 1000,
    'max_epochs': 5,
    'checkpoint_freq': 25,
    'temperature': 0.1,
    'gb_path': 'https://pokemon-ml.s3.amazonaws.com/PokemonRed.gb',
    'init_state': 'https://pokemon-ml.s3.amazonaws.com/has_pokedex_nballs.state',
}

context = ray.init(
    address=address,
    dashboard_port=8265,
    include_dashboard=True,
    runtime_env={
        'working_dir': ".",
        'pip': "C:\\Users\\josep\\Projects\\PlayPokemonRed\\requirements.txt",
        "env_vars": {"PYTHONFAULTHANDLER": "1"},
    },
)
# DataParallelTrainer
trainer = TensorflowTrainer(
    train_loop_per_worker=training_loop,
    scaling_config=ScalingConfig(
        num_workers=1,
        use_gpu=True,
        resources_per_worker={"CPU": 1, "GPU": 1}
    ),
    train_loop_config=training_loop_config,
    run_config=run_config,
)
trainer.fit()
