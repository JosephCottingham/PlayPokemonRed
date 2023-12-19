import os
from typing import Tuple, List
import tensorflow as tf
import numpy as np
import logging
from ..game_env import ACTION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Value_Function(tf.keras.Model):
  def __init__(self, name: str = "Value_Function"):
    super(Value_Function, self).__init__(name=name)
    
    # Accepts the Action data as input
    self.action_d1 = tf.keras.layers.Dense(6,activation='sigmoid', input_shape=(1, 6))
    self.action_d2 = tf.keras.layers.Dense(32,activation='sigmoid')
    self.action_dropout1 = tf.keras.layers.Dropout(0.15)
    self.action_d3 = tf.keras.layers.Dense(64,activation='sigmoid')

    # Accepts the State data as input
    self.c1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(1, 144, 160, 1))
    self.mp1 = tf.keras.layers.MaxPooling2D((2, 2))
    self.flatten = tf.keras.layers.Flatten()
    self.d1 = tf.keras.layers.Dense(512,activation='sigmoid')
    self.dropout1 = tf.keras.layers.Dropout(0.15)
    self.d2 = tf.keras.layers.Dense(256,activation='sigmoid')
    self.dropout2 = tf.keras.layers.Dropout(0.15)
    self.d3 = tf.keras.layers.Dense(64,activation='sigmoid')
    self.dropout3 = tf.keras.layers.Dropout(0.15)
    self.d4 = tf.keras.layers.Dense(32,activation='sigmoid')

    # Merges the two input layers
    self.merge_action_state = tf.keras.layers.Add()

    # Common layers
    self.d4 = tf.keras.layers.Dense(64,activation='sigmoid')
    
    # TODO how many outputs? How does this scale to be comparable with actually reward values?
    self.o = tf.keras.layers.Dense(1,activation='softmax')
  
  def call(self, input_state, input_action):
    # numpy/scalar values in `inputs` get converted to tensors automatically
    input_action = tf.convert_to_tensor(input_action)

    # State data flow
    z = self.c1(input_state)
    z = self.mp1(z)
    z = self.flatten(z)
    z = self.d1(z)
    z = self.dropout1(z)
    z = self.d2(z)
    z = self.dropout2(z)
    z = self.d3(z)

    # Action data flow
    y = self.action_d1(input_action)
    y = self.action_d2(y)
    y = self.action_dropout1(y)
    y = self.action_d3(y)

    # Merged data flow
    merged = self.merge_action_state([z, y])

    # Common layers
    x = self.dropout3(merged)
    return self.o(x)
    

class Policy_Network(tf.keras.Model):
  def __init__(self, name: str = "Policy_Network"):
    super(Policy_Network, self).__init__(name=name)
    initializer = tf.keras.initializers.Zeros()

    # Screen layers
    # kernel size is 16x16 becase the game maps can be broken down into these chunks.
    self.c1 = tf.keras.layers.Conv2D(32, kernel_size=(16, 16), activation='sigmoid', input_shape=(1, 144, 160, 1), kernel_initializer=initializer)
    self.mp1 = tf.keras.layers.AveragePooling2D((4, 4))
    

    self.flatten = tf.keras.layers.Flatten()
    self.d1 = tf.keras.layers.Dense(1024, activation='sigmoid', kernel_initializer=initializer)
    self.dropout1 = tf.keras.layers.Dropout(0.15)
    self.d2 = tf.keras.layers.Dense(512, activation='sigmoid', kernel_initializer=initializer)
    self.dropout2 = tf.keras.layers.Dropout(0.15)
    self.d3 = tf.keras.layers.Dense(256,activation='sigmoid', kernel_initializer=initializer)
    self.dropout3 = tf.keras.layers.Dropout(0.15)
    self.d4 = tf.keras.layers.Dense(128,activation='sigmoid', kernel_initializer=initializer)
    self.o = tf.keras.layers.Dense(6,activation='softmax', kernel_initializer=initializer)

  def call(self, inputs, training=False):
    x = self.c1(inputs)
    x = self.mp1(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.dropout1(x)
    x = self.d2(x)
    x = self.dropout2(x)
    x = self.d3(x)
    x = self.dropout3(x)
    x = self.d4(x)
    return self.o(x)
    
class Agent():
    def __init__(self, name: str = "PPO"):
        self.name = name
        self.policy_network_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.value_function_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.policy_network = Policy_Network()
        self.policy_network.build(input_shape=(1, 144, 160, 1))
        self.value_function = Value_Function()
        self.clip_pram = 0.2
        self.policy_network_losses = []
        self.value_function_losses = []

          
    def act(self, state, temperature=1) -> Tuple[ACTION, np.ndarray]:
      prob = self.policy_network(state)[0]
      prob = np.array(prob)**(1/1-temperature)
      p_sum = prob.sum()
      sample_temp = prob/p_sum 
      return ACTION(np.argmax(np.random.multinomial(1, sample_temp, 1))), prob
    
    def learn(self, states: np.ndarray, actions: List[ACTION],  advantage: np.ndarray, action_probs: np.ndarray, discounted_future_rewards: np.ndarray):
        discounted_future_rewards = tf.reshape(discounted_future_rewards, (len(discounted_future_rewards),))
        adv = tf.reshape(advantage, (len(advantage),))

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.policy_network(states, training=True)
            v = self.value_function(states, np.array([action.numpy() for action in actions], dtype=np.float32), training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discounted_future_rewards, v)
            value_function_loss = 0.5 * tf.keras.metrics.mean_squared_error(discounted_future_rewards, v)
            policy_network_loss = self.calulate_policy_network_loss(p, np.array([action.value for action in actions], dtype=np.int8), adv, tf.convert_to_tensor(action_probs), value_function_loss)
            
        grads1 = tape1.gradient(policy_network_loss, self.policy_network.trainable_variables)
        grads2 = tape2.gradient(value_function_loss, self.value_function.trainable_variables)
        self.policy_network_opt.apply_gradients(zip(grads1, self.policy_network.trainable_variables))
        self.value_function_opt.apply_gradients(zip(grads2, self.value_function.trainable_variables))
        
        self.policy_network_losses.append(policy_network_loss)
        self.value_function_losses.append(value_function_loss)

        logger.info(f"Policy Network Loss:\t{policy_network_loss}")
        logger.info(f"Value Function Loss:\t{value_function_loss}")

        return policy_network_loss, value_function_loss

    def calulate_policy_network_loss(self, probability, actions, adv, action_probs, value_function_loss):
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        #print(probability)
        #print(entropy)
        sur1 = []
        sur2 = []
        
        for prob, t, action_prob, a in zip(probability, adv, action_probs, actions):
          t =  tf.constant(t)
          #op =  tf.constant(op)
          #print(f"t{t}")
          #ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
          ratio = tf.math.divide(prob[a], action_prob[a])
          #print(f"ratio{ratio}")
          s1 = tf.math.multiply(ratio,t)
          #print(f"s1{s1}")
          s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram),t)
          #print(f"s2{s2}")
          sur1.append(s1)
          sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        #closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - value_function_loss + 0.001 * entropy)
        #print(loss)
        return loss

    def save(self, path):
        path = os.path.join(path, f'{self.name}')
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f'model')
        # tf.keras.saving.save_model(
        #   self.policy_network,
        #   path+'_policy_network',
        #   save_format="tf"
        # )
        # tf.keras.saving.save_model(
        #   self.value_function,
        #   path+'_policy_network',
        #   save_format="tf"
        # )
        self.policy_network.save_weights(path+'_policy_network', save_format="tf")
        self.value_function.save_weights(path+'_value_function', save_format="tf")


    @staticmethod
    def load(path: str, name: str):
        agent = Agent(name)
        path = os.path.join(path, f'{name}')
        path = os.path.join(path, f'model')
        agent.policy_network.load_weights(path+'_policy_network')
        agent.value_function.load_weights(path+'_value_function')
        return agent

    def get_metrics(self):
        return {
            'policy_network_losses': self.policy_network_losses,
            'value_function_losses': self.value_function_losses,
        }