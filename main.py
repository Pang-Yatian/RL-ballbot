import gym
import pybullet as p
import numpy as np
from TD3 import Agent
import ballbot
import tensorflow as tf
from collections import deque
import time

train = True
Max_episodes = 100000
best_avg_reward = 0

if train:
    p.connect(p.DIRECT)
else:
    p.connect(p.GUI)

if __name__ == '__main__':
    if train:
        env = gym.make("ballbot-v0")
    else:
        env = gym.make("ballbot_noise-v0")
        #env = gym.make("cublirobot-v0")
    writer = tf.summary.create_file_writer("./mylogs")
    agent = Agent(train=train, env=env)
    total_reward_list = deque(maxlen=50)
    if not train:
        agent.load_model()
        state = env.reset()
        roll = -240
        speed = 0.1
        test_steps = 10000
        total_reward = 0
        time.sleep(1)
        for step in range(test_steps):
            time.sleep(0.0015)
            if (roll > 0 and roll < 360):
                p.resetDebugVisualizerCamera(0.7, 90+roll, -30, [0, 0, 0.1])
            elif roll <= 0:
                p.resetDebugVisualizerCamera(0.7, 90, -30, [0, 0, 0.1])
            else:
                p.resetDebugVisualizerCamera(0.7, 90, -89.9, [0, 0, 0.1])
            roll += speed
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
        print("final reward", total_reward)
    else:
        with writer.as_default():
            for episode in range(Max_episodes):
                state = env.reset()
                done = False
                total_reward = 0
                agent.actor_loss = 0
                agent.critic_loss = 0
                balance_steps = 0
                while not done:

                    action = agent.choose_action(state)
                    next_state, reward, done, info = env.step(action)
                    reward = reward if not done else -1
                    total_reward += reward
                    balance_steps += 1
                    agent.remember(state, action, reward, next_state, done)
                    agent.learn()
                    state = next_state
                total_reward_list.append(total_reward)
                tf.summary.scalar("reward", total_reward, step=episode)
                tf.summary.scalar("steps", balance_steps, step=episode)
                writer.flush()
                if np.mean(total_reward_list) > best_avg_reward:
                    agent.save_model()
                    best_avg_reward = np.mean(total_reward_list)
                print("episode:", episode, "score", total_reward)
