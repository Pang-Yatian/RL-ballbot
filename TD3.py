import tensorflow as tf
import numpy as np
from Replay_Buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, train, alpha=0.001, beta=0.002, env=None, gamma=0.9, buffer_size=1000000, batch_size=5096,
                 update=5, tau=0.005):
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.train = train
        self.actor_loss = 0
        self.critic_loss = 0
        self.learn_counter = 0
        self.update_fre = update

        self.actor_model = ActorNetwork(name='actor')
        self.target_actor_model = ActorNetwork(name='target_actor')
        self.critic_model1 = CriticNetwork(name='critic1')
        self.target_critic_model1 = CriticNetwork(name='target_critic1')
        self.critic_model2 = CriticNetwork(name='critic2')
        self.target_critic_model2 = CriticNetwork(name='target_critic2')

        self.actor_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.target_actor_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha))
        self.critic_model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.target_critic_model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.critic_model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))
        self.target_critic_model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=beta))

        self.targetnet_update(self.actor_model, self.target_actor_model)
        self.targetnet_update(self.critic_model1, self.target_critic_model1)
        self.targetnet_update(self.critic_model2, self.target_critic_model2)

    def remember(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def save_model(self):
        print("------saving model------")
        self.actor_model.save_weights(self.actor_model.checkpoint_file)
        self.critic_model1.save_weights(self.critic_model1.checkpoint_file)
        self.critic_model2.save_weights(self.critic_model2.checkpoint_file)

    def load_model(self):
        print("------loading model------")
        self.actor_model.load_weights(self.actor_model.checkpoint_file)
        self.critic_model1.load_weights(self.critic_model1.checkpoint_file)
        self.critic_model2.load_weights(self.critic_model2.checkpoint_file)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        action = self.actor_model.add_noise(state)
        return action[0]

    def targetnet_update(self,net,target_net):
        for a, b in zip(net.variables, target_net.variables):
            b.assign(a)

    def fresh_targetnet(self):
        for a, b in zip(self.actor_model.variables, self.target_actor_model.variables):
            new_actor_variables = (1-self.tau) * b + self.tau * a
            b.assign(new_actor_variables)
        for c, d in zip(self.critic_model1.variables, self.target_critic_model1.variables):
            new_critic_variables = (1-self.tau) * d + self.tau * c
            d.assign(new_critic_variables)
        for e, f in zip(self.critic_model2.variables, self.target_critic_model2.variables):
            new_critic_variables = (1-self.tau) * f + self.tau * e
            f.assign(new_critic_variables)
        #print("******target net refreshed******")

    def learn(self):
        if len(self.memory.Memory) <= self.batch_size:
            return
        if self.learn_counter == 10000:
            self.learn_counter = 0
        self.learn_counter += 1

        batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
            self.memory.sample(batch_size=self.batch_size)
        next_new_action = self.target_actor_model.add_noise(batch_next_state)
        target_q1 = tf.squeeze(self.target_critic_model1.call(batch_next_state, next_new_action), 1)
        target_q2 = tf.squeeze(self.target_critic_model2.call(batch_next_state, next_new_action), 1)
        target_q_min = tf.minimum(target_q1, target_q2)

        target_q_value = batch_reward + (1-batch_done) * self.gamma * target_q_min

        with tf.GradientTape() as q1_tape:
            predicted_q1 = tf.squeeze(self.critic_model1.call(batch_state, batch_action), 1)
            q1_loss = tf.reduce_mean(tf.square(predicted_q1 - target_q_value))
        q1_grad = q1_tape.gradient(q1_loss,self.critic_model1.variables)
        self.critic_model1.optimizer.apply_gradients(zip(q1_grad, self.critic_model1.variables))

        with tf.GradientTape() as q2_tape:
            predicted_q2 = tf.squeeze(self.critic_model2.call(batch_state, batch_action), 1)
            q2_loss = tf.reduce_mean(tf.square(predicted_q2 - target_q_value))
        q2_grad = q2_tape.gradient(q2_loss,self.critic_model2.variables)
        self.critic_model2.optimizer.apply_gradients(zip(q2_grad, self.critic_model2.variables))

        if self.learn_counter % self.update_fre == 0:
            with tf.GradientTape() as p_tape:
                new_action = self.actor_model.call(batch_state)
                predicted_new_q = self.critic_model1.call(batch_state,new_action)
                p_loss = -tf.reduce_mean(predicted_new_q)
            p_grad = p_tape.gradient(p_loss, self.actor_model.variables)
            self.actor_model.optimizer.apply_gradients(zip(p_grad, self.actor_model.variables))

            self.fresh_targetnet()


