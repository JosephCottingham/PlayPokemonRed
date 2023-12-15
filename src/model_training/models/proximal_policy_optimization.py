import tensorflow as tf
import numpy as np

from ..game_env import ACTION

class Critic(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.c1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(1, 144, 160, 1))
    self.mp1 = tf.keras.layers.MaxPooling2D((2, 2))
    self.flatten = tf.keras.layers.Flatten()
    self.d1 = tf.keras.layers.Dense(512,activation='sigmoid')
    self.dropout1 = tf.keras.layers.Dropout(0.15)
    self.d2 = tf.keras.layers.Dense(256,activation='sigmoid')
    self.dropout2 = tf.keras.layers.Dropout(0.15)
    self.d3 = tf.keras.layers.Dense(64,activation='sigmoid')
    self.dropout3 = tf.keras.layers.Dropout(0.15)
    self.o = tf.keras.layers.Dense(1,activation='softmax')
  def call(self, input_data):
    x = self.c1(input_data)
    x = self.mp1(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.dropout1(x)
    x = self.d2(x)
    x = self.dropout2(x)
    x = self.d3(x)
    x = self.dropout3(x)
    x = self.o(x)
    return x
    

class Actor(tf.keras.Model):
  def __init__(self):
    super().__init__()
    initializer = tf.keras.initializers.Zeros()

    self.c1 = tf.keras.layers.Conv2D(32, kernel_size=(4, 4), activation='sigmoid', input_shape=(1, 144, 160, 1), kernel_initializer=initializer)
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

  def call(self, input_data):
    x = self.c1(input_data)
    x = self.mp1(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.dropout1(x)
    x = self.d2(x)
    x = self.dropout2(x)
    x = self.d3(x)
    x = self.dropout3(x)
    x = self.d4(x)
    x = self.o(x)
    return x
    
class Agent():
    def __init__(self):
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.actor = Actor()
        self.critic = Critic()
        self.clip_pram = 0.2

          
    def act(self, state, temperature=1):
      prob = self.actor(state)[0]
      prob = np.array(prob)**(1/1-temperature)
      p_sum = prob.sum()
      sample_temp = prob/p_sum 
      return ACTION(np.argmax(np.random.multinomial(1, sample_temp, 1))), prob

    # def act(self,state):
    #     prob = self.actor(state)[0]
    #     return ACTION(np.argmax(prob)), prob

    def learn(self, states, actions,  adv , old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p),6))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * tf.keras.metrics.mean_squared_error(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)
            
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss

    def actor_loss(self, probs, actions, adv, old_probs, closs):
        probability = probs      
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        #print(probability)
        #print(entropy)
        sur1 = []
        sur2 = []
        
        for pb, t, op, a in zip(probability, adv, old_probs, actions):
          t =  tf.constant(t)
          #op =  tf.constant(op)
          #print(f"t{t}")
          #ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
          ratio = tf.math.divide(pb[a],op[a])
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
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        #print(loss)
        return loss