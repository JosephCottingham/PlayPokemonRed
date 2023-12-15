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
        'act_freq': 16,
        'window_type': 'SDL2', #SDL2
        'disable_input': True,
    },
    'max_steps': 60,
    'temperature': 0.1,
    'gb_path': 'https://pokemon-ml.s3.amazonaws.com/PokemonRed.gb',
    'init_state': 'https://pokemon-ml.s3.amazonaws.com/has_pokedex_nballs.state',
}
ray.init()

trainer = TensorflowTrainer(
    train_loop_per_worker=training_loop,
    scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    train_loop_config=training_loop_config
)
trainer.fit()
