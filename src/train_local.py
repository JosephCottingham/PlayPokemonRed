import ray
from ray.runtime_env import RuntimeEnv
from ray.train.tensorflow import TensorflowTrainer
from ray.train import ScalingConfig

from model_training.training_loop import training_loop

# Single worker with a CPU
scaling_config = ScalingConfig(num_workers=1, use_gpu=False)

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
    'max_episodes': 20,
    'max_epochs': 150,
    'checkpoint_freq': 10,
    'temperature': 0.1,
    'gb_path': 'https://pokemon-ml.s3.amazonaws.com/PokemonRed.gb',
    'init_state': 'https://pokemon-ml.s3.amazonaws.com/has_pokedex_nballs.state',
}

training_loop(training_loop_config)