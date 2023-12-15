import time
import os
import requests
import logging

import tensorflow as tf
import numpy as np

from .game_env import Game_Env
from .models.proximal_policy_optimization import Agent

def preprocess1(states, actions, rewards, done, values, gamma):
    g = 0
    lmbda = 0.95
    returns = []
    for i in reversed(range(len(rewards))):
       delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
       g = delta + gamma * lmbda * done[i] * g
       returns.append(g + values[i])

    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - values[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    returns = np.array(returns, dtype=np.float32)
    return states, actions, returns, adv 

def download_file(url, filename):
    logging.info(f"Downloading {url} to {filename}")
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    return filename


def training_loop(training_loop_config: dict) -> str:
    training_loop_config['env_config']['gb_path'] = download_file(training_loop_config['gb_path'], 'rom.gb')
    training_loop_config['env_config']['init_state'] = download_file(training_loop_config['init_state'], 'init.state')

    # Create Environment
    env = Game_Env(training_loop_config['env_config'], num_env=0)

    state = env.get_current_state()
    
    # Session UUID for which all data will be related to
    session_uuid = env.session_uuid

    prev_step_reward = 0
    prev_state = state
    steps = 0
    file1 = open(os.path.join(env.session_dir_path, 'checkpoint_log.csv'), "w")
    file1.write(f"Step,Starting Reward,Ending Reward,Reward Diff,battle type\n")
    file1.close()

    # Create Agent
    agent = Agent()

    while steps < training_loop_config['max_steps']:
        steps += 1
        print(f"Step: {steps}")

        done = False
        rewards = []
        states = []
        actions = []
        probs = []
        dones = []
        values = []
        print("new episod")
        
        e = 0
        while e < 250 and prev_state.battle_type == state.battle_type:
            e+=1
            print(f'State: {state.get().shape}')
            action, prob = agent.act(state.get(), temperature=training_loop_config['temperature'])
            value = agent.critic(state.get()).numpy()

            # state = env.step(action)
            print(f'Action: {action.name} | Step: {e} | Prob: {prob}')
            next_state = env.step(action)
            next_state, reward, done, _ = env.step(action)
            print(reward)

            dones.append(1-done)
            rewards.append(reward.get())
            states.append(state.get_without_element_dim())
            actions.append(action.value)
                        
            probs.append(prob)
            values.append(value[0][0])
            prev_state = state
            state = next_state


        # Train Model
        print('Run Training')
        value = agent.critic(state.get()).numpy()
        values.append(value[0][0])
        np.reshape(probs, (len(probs),6))
        probs = np.stack(probs, axis=0)

        states, actions, returns, adv  = preprocess1(states, actions, rewards, dones, values, 1)



        al, cl = agent.learn(states, actions, adv, probs, returns)
        print(f"Actor Loss:\t{al}") 
        print(f"Critic Loss:\t{cl}")

        if steps%15 == 0:
            # Save Model Checkpoint
            file1 = open(os.path.join(env.session_dir_path, 'checkpoint_log.csv'), "a")
            file1.write(f"{steps},{prev_step_reward},{reward.get()},{reward.get()-prev_step_reward},{prev_state.battle_type.name}\n")
            file1.close()
            prev_step_reward=reward.get()

            print('Save Model Checkpoint')
            model_actor_path = os.path.join(env.session_dir_path, 'model_checkpoint', 'model_actor_{}'.format(steps))
            model_critic_path = os.path.join(env.session_dir_path, 'model_checkpoint', 'model_critic_{}'.format(steps))

            agent.actor.save(model_actor_path, save_format="tf")
            agent.critic.save(model_critic_path, save_format="tf")

            env.save_game('game_save_{}'.format(steps))
            env.reset()
    return session_uuid