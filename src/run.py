from game_env import Game_Env
import time
import os
import tensorflow as tf
import numpy as np
from POO.POO import agent
from train import train
import multiprocessing
   

if __name__ == '__main__':

    tf.random.set_seed(336699)
    agent = agent()
    # agent.actor = tf.keras.saving.load_model(os.path.join("../sessions/5a8fac14-f106-4005-adee-deb08169399c/model_checkpoint/model_actor_1"))
    # agent.critic = tf.keras.saving.load_model(os.path.join("../sessions/5a8fac14-f106-4005-adee-deb08169399c/model_checkpoint/model_critic_1"))

    max_steps = 60
    process_count=2
    processes=[]

    env_config = {
        'gb_path': '../ROMs/PokemonRed.gb',
        'init_state': '../ROMs/has_pokedex_nballs.state',
        'save_screenshots': False,
        'act_freq': 16,
        'window_type': 'SDL2', #SDL2
        'disable_input': True,
    }
    # run train as proccess
    pool = multiprocessing.Pool(processes = 3)
    args = [(procnum, env_config, max_steps) for procnum in range(process_count)]
    session_uuids = pool.starmap(train, args)
    print(session_uuids)