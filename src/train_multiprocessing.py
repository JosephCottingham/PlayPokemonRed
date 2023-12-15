from model_training.game_env import Game_Env
import time
import os
import tensorflow as tf
import numpy as np
from model_training.models.proximal_policy_optimization import agent
from model_training.training_loop import training_loop
import multiprocessing
   

if __name__ == '__main__':

    tf.random.set_seed(336699)
    agent = agent()
    # agent.actor = tf.keras.saving.load_model(os.path.join("../sessions/5a8fac14-f106-4005-adee-deb08169399c/model_checkpoint/model_actor_1"))
    # agent.critic = tf.keras.saving.load_model(os.path.join("../sessions/5a8fac14-f106-4005-adee-deb08169399c/model_checkpoint/model_critic_1"))

    max_steps = 60
    process_count=2
    processes=[]

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
    # run train as proccess
    pool = multiprocessing.Pool(processes = 1)
    args = [(training_loop_config,) for procnum in range(process_count)]
    session_uuids = pool.starmap(training_loop, args)
    print(session_uuids)