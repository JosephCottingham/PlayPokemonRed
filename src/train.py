from game_env import Game_Env
import time
import os
import tensorflow as tf
import numpy as np
from POO.POO import agent

def make_env(rank, env_conf, seed=0, num_env=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = Game_Env(env_conf, num_env=num_env)
        #env.seed(seed + rank)
        return env
    return _init


def preprocess1(states, actions, rewards, done, values, gamma):
    g = 0
    lmbda = 0.95
    returns = []
    for i in reversed(range(len(rewards))):
       delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
       g = delta + gamma * lmbda * dones[i] * g
       returns.append(g + values[i])

    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - values[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    returns = np.array(returns, dtype=np.float32)
    return states, actions, returns, adv 

def train(procnum: int, env_config: dict, max_steps: int) -> str:
    num_cpu = 1 #64 #46  # Also sets the number of episodes per training iteration
    env = make_env(0, env_config, num_env=procnum)()
    state = env.get_current_state()
    session_uuid = env.session_uuid

    prev_step_reward = 0
    prev_state = state
    steps = 0
    file1 = open(os.path.join(env.session_dir_path, 'checkpoint_log.csv'), "w")
    file1.write(f"Step,Starting Reward,Ending Reward,Reward Diff,battle type\n")
    file1.close()


    while steps < max_steps:
        steps += 1
        print(f"Step: {steps}")

        done = False
        all_aloss = []
        all_closs = []
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
            action, prob = agent.act(state.get(), temperature=0.1)
            value = agent.critic(state.get()).numpy()

            # state = env.step(action)
            print(f'Action: {action.name} | Step: {e} | Prob: {prob}')
            next_state = env.step(action)
            next_state, reward, done, _ = env.step(action)
            print(reward)

            dones.append(1-done)
            rewards.append(reward.get())
            states.append(state.get_without_element_dim())
            #actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
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



        al,cl = agent.learn(states, actions, adv, probs, returns)
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