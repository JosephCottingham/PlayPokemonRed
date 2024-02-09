import ray
from ray.runtime_env import RuntimeEnv
from ray.train.tensorflow import TensorflowTrainer
from ray.train import ScalingConfig

from model_training.play_loop import play_loop

# Single worker with a CPU
scaling_config = ScalingConfig(num_workers=1, use_gpu=False)

loop_config = {
    'env_config': {
        'save_screenshots': False,
        'save_new_state_screen_shot': False,
        'tick_freq': 60,
        'emulation_speed': 5,
        'window_type': 'SDL2', #SDL2
        'disable_input': True,
    },
    'screen_shot_freq': 10,
    'temperature': 0.1,
    'gb_path': 'https://pokemon-ml.s3.amazonaws.com/PokemonRed.gb',
    'init_state': 'https://pokemon-ml.s3.amazonaws.com/has_pokedex_nballs.state',
    'checkpoint_dir': 's3://pokemon-ml/checkpoints/LearnPokemonRed/TensorflowTrainer_b2a10_00000_0_2024-01-14_16-26-40/checkpoint_000008/'
}

play_loop(loop_config)