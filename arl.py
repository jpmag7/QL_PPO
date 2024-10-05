from QLearning import QLearningAgent
import torch
import torch.nn as nn
import random


# Open dataset
datafile = open("shakespeare.txt", "r")
rawdata = datafile.read()
datafile.close()
datasize = len(rawdata)

# Possoble tokens
vocab = sorted(list(set(rawdata)))

# Build encoder and decoder
stoi = {ch:i for i,ch in enumerate(vocab)}
itos = {i:ch for i,ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

encoded_data = encode(rawdata)

block_size = 32

def get_train_sample():
	start  = random.randint(0, datasize - block_size - 1)
	end    = random.randint(start + 1, start + block_size)
	sample = encoded_data[start:end]
	sample = [-1] * (block_size - len(sample)) + sample
	return sample, encoded_data[end]


# Build the models
generator = QLearningAgent(
	block_size,
	len(vocab),
	hidden_sizes=[32, 32, 32],
	learning_rate=0.01,
	discount_factor=0.9,
	exploration_rate=0.05
)
descriminator = QLearningAgent(
	block_size+1,
	2,
	hidden_sizes=[16, 16, 16],
	learning_rate=0.01,
	discount_factor=0.9,
	exploration_rate=0
)


# Training cicle
it = 100000
for c in range(it):
	for _ in range(10):
		state, res = get_train_sample()
		gen_action = generator.select_action(state)
		next_real_state = state + [res]
		next_pred_state = state + [gen_action]

		if random.randint(0,1) == 0:
			# Real state has to output 1
			des_action = descriminator.select_action(next_real_state)
			des_reward = 1 if des_action == 1 else -1
			descriminator.update([(next_real_state, des_action, des_reward, next_real_state)])
			# Pred state has to output 0
			des_action = descriminator.select_action(next_pred_state)
			des_reward = 1 if des_action == 0 else -1
			descriminator.update([(next_pred_state, des_action, des_reward, next_pred_state)])
		else:
			# Pred state has to output 0
			des_action = descriminator.select_action(next_pred_state)
			des_reward = 1 if des_action == 0 else -1
			descriminator.update([(next_pred_state, des_action, des_reward, next_pred_state)])
			# Real state has to output 1
			des_action = descriminator.select_action(next_real_state)
			des_reward = 1 if des_action == 1 else -1
			descriminator.update([(next_real_state, des_action, des_reward, next_real_state)])

	state, res = get_train_sample()
	gen_action = generator.select_action(state)
	next_pred_state = state + [gen_action]
	gen_reward = 1 if descriminator.select_action(next_pred_state) == 1 else -1
	generator.update([(state, gen_action, gen_reward, next_pred_state[1:])])
	if c % 1000 == 0:
		print("#######\nState:", decode(list(filter(lambda x: x != -1, state))))
		print("Gen predicted:", decode([gen_action]), "Res:", decode([res]))
		print("Gen_reward:", gen_reward)
		print("Des_reward", des_reward)