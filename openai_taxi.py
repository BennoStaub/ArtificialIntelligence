import gym
from IPython.display import clear_output
from time import sleep
import os
import numpy as np
import random
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Embedding, Reshape
from keras.optimizers import Adam



class RandomAgent():

	def __init__(self, env):
		self.env = env

	def run(self):
		self.frames = []
		done = False
		steps = 0

		while not done:
			action = self.env.action_space.sample()
			state, reward, done, info = self.env.step(action)

			self.frames.append({
				'frame': self.env.render(mode='ansi'),
				'state': state,
				'action': action,
				'reward': reward
				}
			)
			steps += 1
			print('Step ' + str(steps))

class QLearningAgent():
	def __init__(self, env, alpha, gamma, epsilon):
		self.env = env
		self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon

	def run(self, epochs):
		for i in range(1, epochs + 1):
			if i == epochs:
				self.frames = []
			state = self.env.reset()
			steps = 0
			done = False

			while not done:
				if random.uniform(0, 1) < self.epsilon:
					action = self.env.action_space.sample() # Explore action space
				else:
					action = np.argmax(self.q_table[state]) # Exploit learned values
				next_state, reward, done, info = self.env.step(action) 
		        
				old_value = self.q_table[state, action]
				next_max = np.max(self.q_table[next_state])

				new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
				self.q_table[state, action] = new_value

				if i == epochs:
					self.frames.append({
						'frame': self.env.render(mode='ansi'),
						'state': state,
						'action': action,
						'reward': reward
						}
					)

				state = next_state
				steps += 1
			print('Epoch ' + str(i) + ' took ' + str(steps))

class DeepQAgent():
	def __init__(self, env, memory_limit=50000):
		self.env = env
		self.model = Sequential()
		self.model.add(Embedding(500, 10, input_length = 1))
		self.model.add(Reshape((10,)))
		self.model.add(Dense(50, activation='relu'))
		self.model.add(Dense(50, activation='relu'))
		#self.model.add(Dense(50, activation='relu'))
		self.model.add(Dense(self.env.action_space.n, activation='linear'))
		print(self.model.summary)
		self.memory = SequentialMemory(limit=memory_limit, window_length=1)
		self.policy = EpsGreedyQPolicy()
		self.agent = DQNAgent(model=self.model, nb_actions=self.env.action_space.n, memory=self.memory, nb_steps_warmup=500, target_model_update=1e-2, policy=self.policy)
		self.agent.compile(Adam(lr=1e-3), metrics=['mae'])

	def fit(self, nb_steps=1000000):
		self.agent.fit(self.env, nb_steps=nb_steps, visualize=False, verbose=1, nb_max_episode_steps=99, log_interval=10000)

	def test(self, nb_episodes=5):
		self.agent.test(self.env, nb_episodes=nb_episodes, visualize=True, nb_max_episode_steps=2000)



def print_frames(frames):
	for i, frame in enumerate(frames):
		os.system('clear')
		print(frame['frame'])
		print("Timestep: " + str(i + 1))
		print("State: " + str(frame['state']))
		print("Action: " + str(frame['action']))
		print("Reward: " + str(frame['reward']))
		sleep(.5)


env = gym.make("Taxi-v2").env
env.reset()
#agent = RandomAgent(env)
#agent = QLearningAgent(env, .1, .6, .1)
#agent.run(10000)
agent = DeepQAgent(env)
agent.fit()
print('fitted')
sleep(3)
agent.test()


#print_frames(agent.frames)
