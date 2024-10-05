import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PPOAgent:
    def __init__(self, actor_nn, critic_nn, learning_rate, clip_ratio, gamma, lamda):
        self.actor_nn = actor_nn
        self.critic_nn = critic_nn
        self.learning_rate = learning_rate
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lamda = lamda
        self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)

    def choose_action(self, state):
        action_probs = self.actor_nn(state)
        #action_probs = tf.nn.softmax(logits)
        action = np.random.choice(range(len(action_probs.numpy()[0])), p=action_probs.numpy()[0])
        return action, tf.math.log(action_probs[0, action])

    def train(self, states, actions, advantages, returns, old_action_probs):
        with tf.GradientTape() as tape:
            # Calculate actor and critic losses
            new_action_probs = self.actor_nn(states)
            values = self.critic_nn(states)
            #new_action_probs = tf.nn.softmax(logits)
            new_action_log_probs = tf.math.log(new_action_probs + 1e-10)
            entropy = -tf.reduce_mean(new_action_probs * new_action_log_probs)

            ratio = tf.exp(new_action_log_probs - old_action_probs)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            critic_loss = tf.reduce_mean(tf.square(returns - values))

            total_loss = actor_loss - 0.5 * critic_loss + 0.01 * entropy

        # Calculate gradients and update weights
        gradients = tape.gradient(total_loss, self.actor_nn.trainable_variables + self.critic_nn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor_nn.trainable_variables + self.critic_nn.trainable_variables))

    def get_advantages_and_returns(self, rewards, values, dones, next_value):
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        last_advantage = 0

        if not dones[-1]:
            returns[-1] = values[-1] + self.gamma * next_value
            last_advantage = self.gamma * (next_value - values[-1])

        for t in reversed(range(len(rewards) - 1)):
            mask = 1 - dones[t]
            returns[t] = rewards[t] + self.gamma * returns[t+1] * mask
            td_error = rewards[t] + self.gamma * values[t+1] * mask - values[t]
            advantages[t] = td_error + self.gamma * self.lamda * last_advantage * mask
            last_advantage = advantages[t]

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        return advantages, returns
