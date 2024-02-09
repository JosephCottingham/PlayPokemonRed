from argparse import Action
from audioop import avg
from datetime import datetime
import logging
import os
from typing import Tuple, List
import tempfile
import json
import shutil

import numpy as np

from ray import train
from ray.train import Checkpoint, ScalingConfig

from model_training.models.proximal_policy_optimization import Agent, Policy_Network, Value_Function
from model_training.game_env import Game_Env, BATTLE_TYPE, ACTION, State, Reward
import json

logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger("ray.train")


class Training_Data:
    '''
    Class for storing training data
    '''
    states: List[State] = []
    rewards: List[float] = []
    actions: List[ACTION] = []
    action_probs: List[np.ndarray] = []
    q_values: List[np.float64] = []

    def __init__(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.action_probs = []
        self.q_values = []

    def get_prev_screens(self) -> List[np.ndarray]:
        return [state.current_screen for state in self.states[-2:]]

    def add(self, states: State, reward: float, action: ACTION, action_prob: np.ndarray, q_values: np.float64):
        self.states.append(states)
        self.rewards.append(reward)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.q_values.append(q_values)

    def add_q_value(self, q_value: np.float64):
        self.q_values.append(q_value)

    def is_empty(self):
        return len(self.actions) == 0

    def get_gae(self) -> Tuple[List[float], np.ndarray]:
        '''
        Generates and returns the generalized advantage estimation

        Returns:
            gae_values: np.ndarray
            adv_values: np.ndarray
        '''
        logger.info("Calculating GAE")
        # Only the last action ends the episode. This is set to zero to invalidate the final q value.
        episode_complete = np.ones(len(self.actions))
        episode_complete[-1] = 0

        # Discount Factor
        GAMMA = 0.99

        # Smoothing Parameter that reduces the variance of the estimate
        # In practice we set 0 < λ < 1 to control the compromise between bias and variance
        # A value of 0 corresponds to the TD(0) (Temporal Difference: only using actual rewards) and has high bias
        # a value of 1 corresponds to vanilla policy gradient with a baseline and has high variance
        LAMBDA = 0.95

        gae = 0
        gae_values = []

        # Calculate GAE (GAE values represent the discounted future rewards)
        for i in reversed(range(len(self.rewards))):
            # calculates the temporal difference error
            delta = self.rewards[i] + GAMMA * self.q_values[i + 1] * episode_complete[i] - self.q_values[i]
            gae = delta + GAMMA * LAMBDA * episode_complete[i] * gae
            gae_values.insert(0, gae + self.q_values[i])


        # Get advantage values
        # advantage values represent the difference between the actual future rewards and the baseline estimate.
        # advantage = Actual Future Reward (Discounted Rewards) - Q Value (Baseline Estimate)
        gae_values = np.array(gae_values, dtype=np.float32)
        q_values = np.array(self.q_values[:-1], dtype=np.float32)
        adv_values = gae_values - q_values

        # Standardizing advantage values to a mean of 0 and std of 1
        adv_values = (adv_values - np.mean(adv_values)) / (np.std(adv_values) + 1e-10)
        
        logger.info(f"AVG GAE Values: {np.mean(gae_values)}")

        return gae_values, adv_values



class Controller:
    '''
    Controller class for training, testing, and running the model.
    '''
    env: Game_Env
    training_loop_config: dict

    models = {
        'COMBAT': Agent(name='COMBAT'),
        'NON_COMBAT': Agent(name='NON_COMBAT')
    }

    def __init__(self, env: Game_Env, training_loop_config: dict, checkpoint: Checkpoint, process_ids: dict = None):
        '''
        Constructor for Controller class.
        '''
        self.env = env
        self.training_loop_config = training_loop_config
        self.temperature = training_loop_config.get('temperature', 1)
        self.process_ids = process_ids
        self.screen_shot_freq = training_loop_config.get('screen_shot_freq')
        self.screenshot_dir = tempfile.mkdtemp()
        self.screenshots_s3_path = training_loop_config.get('screenshots_s3_path')
        self.screenshots_s3_bucket = training_loop_config.get('screenshots_s3_bucket')
        
        self.total_epoch = 0
        
        logger.info(f"Screenshot dir: {self.screenshot_dir}")

        if self.training_loop_config.get('checkpoint_dir'):
            logger.info(f"Loading checkpoint: {self.training_loop_config['checkpoint_dir']}")
            for key, model in self.models.items():
                logger.info(f"Loaded model: {key}") 
                model.load(self.training_loop_config['checkpoint_dir'], key)
            self.checkpoint = checkpoint
            self.prev_checkpoint_metadata = {}
            if checkpoint:
                self.prev_checkpoint_metadata = checkpoint.get_metadata()

    def play(self):
        '''
        Plays the game with the model
        '''
        logger.info("Playing game")

        while True:
            cur_state = self.env.get_current_state()
            # Get action from model
            action, prob = self.models[self.get_model_key(cur_state.battle_type)].act(cur_state.get(), temperature=self.temperature)
            logger.debug(f"Performing action: {action.name}")
            _ = self.env.step(action, cur_state)

    def train(self):
        '''
        Trains the models
        '''
        logger.info("Training models")
        
        episode=0
        cur_state=None
        prev_state=None

        while episode < self.training_loop_config['max_episodes']:
            episode+=1
            logger.info(f"Excuting Episode: {episode} in checkpoint reload: {self.training_loop_config['checkpoint_reload']}")
            epoch = 0
            training_data = Training_Data()
            now = datetime.now()
            while epoch < self.training_loop_config['max_epochs']:
                epoch+=1
                logger.debug(f"Excuting Epoch: {epoch}")
                cur_state = self.env.get_current_state()
                battle_type = None
                # If the battle type changed, episode is over
                if prev_state is None or prev_state.battle_type == cur_state.battle_type:
                    cur_state.set_prev_screens(training_data.get_prev_screens())
                    if self.screen_shot_freq and epoch % self.screen_shot_freq == 0:
                        cur_state.save_screenshot(self.screenshot_dir, self.screenshots_s3_bucket, self.screenshots_s3_path, True)
                    
                    reward, action, prob, q_value = self.collect_training_data(cur_state)
                    logger.debug(f"Reward: {reward.get_new_reward()}")
                    training_data.add(cur_state, reward.get_new_reward(), action, prob, q_value)
                    prev_state = cur_state
                    battle_type = prev_state.battle_type
                else:
                    logger.info(f"Battle type changed, ending step: {prev_state.battle_type} -> {cur_state.battle_type}")
                    # Due to change in battle type, we need to add the last state to the training data and break out of the epoch loop.
                    battle_type = prev_state.battle_type
                    prev_state = cur_state
                    break
            self.total_epoch += epoch
            if training_data.is_empty():
                logger.info("No training data collected, skipping training likly due to battle type change")
                continue

            # Add final q value, this value is never used due to implimnetation in get_gae but is required to prevent an out of bounds error.
            training_data.add_q_value(np.float64(0))

            # Note that PyGame is paused during training, as we are not call tick on the emulator.
            self.fit_model(self.get_model_key(battle_type), training_data)
            time_taken_for_episode = datetime.now() - now
            logger.info(f"Episode {episode} took {time_taken_for_episode}")

            # Save check point at the end of training if freq matches.
            if (episode != 0 and episode % self.training_loop_config['checkpoint_freq'] == 0) or episode == self.training_loop_config['max_episodes']:
                now = datetime.now()
                datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"checkpoint_{episode}_{datetime_str}"
                self.save_checkpoint(episode, checkpoint_name)
                time_taken_for_checkpoint = datetime.now() - now
                logger.info(f"Checkpoint {checkpoint_name} took {time_taken_for_checkpoint}")
            else:
                self.save_metrics_only(episode)

        logger.info(f"Training completed {episode} Episodes")

    def collect_training_data(self, cur_state: State) -> Tuple[Reward, ACTION, np.ndarray, np.float64]:
        '''
        Collects training data
        '''
        logger.debug("Collecting training data")
        # Get action from model
        action, prob = self.models[self.get_model_key(cur_state.battle_type)].act(cur_state.get(), temperature=self.temperature)
        q_value = self.get_q_value(cur_state, action)
        
        logger.debug(f"Performing action: {action.name}")
        reward = self.env.step(action, cur_state)
        
        return reward, action, prob, q_value

    def get_q_value(self, cur_state: State, action: ACTION) -> np.float64:
        return self.models[self.get_model_key(cur_state.battle_type)].value_function(cur_state.get(), np.reshape(action.numpy(), (1,6))).numpy()[0][0]


    def fit_model(self, model_key: str, training_data: Training_Data):
        '''
        Fits a model
        '''
        logger.info(f"Fitting model: {model_key}")
        gae_values, adv_values = training_data.get_gae()

        total_record_count = len(training_data.states)
        logger.info(f"Training data size: {total_record_count}")
        for start in range(0, total_record_count, 100):
            # logger.info(f"Training batch: {start}")
            end = start + 100
            if end > total_record_count:
                end = total_record_count

            logger.info(f"Training batch: {start} - {end}")
            
            policy_network_loss, value_function_loss = self.models[model_key].learn(
                states=np.array(
                    [state.get_without_element_dim() for state in training_data.states[start:end]],
                    dtype=np.float32
                ),
                actions=training_data.actions[start:end],
                advantage=adv_values[start:end],
                action_probs=np.array(training_data.action_probs[start:end], dtype=np.float32),
                discounted_future_rewards=np.array(gae_values[start:end], dtype=np.float32)
            )


    def get_model_key(self, battle_type):
        return 'NON_COMBAT' if battle_type == BATTLE_TYPE.NON_COMBAT else 'COMBAT'

    def save_checkpoint(self, episode: int, checkpoint_name: str):
        '''
        Saves the checkpoint
        '''
        logger.info(f"Saving checkpoint: {checkpoint_name}")
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:

            # copy screenshot dir to temp dir and clear
            shutil.copytree(self.screenshot_dir, os.path.join(temp_checkpoint_dir, 'screenshots'))
            shutil.rmtree(self.screenshot_dir, ignore_errors=True)
            os.makedirs(self.screenshot_dir, exist_ok=True)

            model_count = 0
            avg_policy_network_loss = 0

            self.env.save(temp_checkpoint_dir)

            metrics = {}  
            for key, model in self.models.items():
                model.save(temp_checkpoint_dir)
                metrics[key] = model.get_metrics()
                model_count += 1
                avg_policy_network_loss += np.mean(metrics[key]['policy_network_losses'])

            avg_policy_network_loss=avg_policy_network_loss/model_count
            metrics['avg_policy_network_loss'] = avg_policy_network_loss
            
            total_reward = self.env.get_total_reward()
            metrics['total_reward'] = total_reward

            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            metadata = {
                "checkpoint_reloads": self.prev_checkpoint_metadata.get('checkpoint_reloads', 0) + 1,
                "episode": episode,
                "epoch_count": self.total_epoch,
                "prev_epoch_count": self.prev_checkpoint_metadata.get('total_epoch_count', 0),
                "total_epoch_count": self.total_epoch + self.prev_checkpoint_metadata.get('total_epoch_count', 0),
                "reward": total_reward,
                "avg_policy_network_loss": avg_policy_network_loss,
                "previous_checkpoint_path": self.checkpoint.path,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            metadata.update(self.process_ids)

            checkpoint.set_metadata(metadata)
            train.report(metrics, checkpoint=checkpoint)
    
    def save_metrics_only(self, episode: int):
        metrics = {}
        model_count = 0
        avg_policy_network_loss = 0

        for key, model in self.models.items():
            metrics[key] = model.get_metrics()
            model_count += 1
            avg_policy_network_loss += np.mean(metrics[key]['policy_network_losses'])

        avg_policy_network_loss=avg_policy_network_loss/model_count
        total_reward = self.env.get_total_reward()

        metrics['avg_policy_network_loss'] = avg_policy_network_loss
        metrics['total_reward'] = total_reward

        train.report(metrics)