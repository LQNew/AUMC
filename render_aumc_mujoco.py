"""Visualization of the MuJoCo environments when loading trained policy model."""
import numpy as np
import random
import torch
import gym
import argparse
import os

from DDPG import DDPG_aumc
from TD3 import TD3_aumc
from SAC import SAC_aumc

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="DDPG_aumc", type=str)          # Policy name
	parser.add_argument("--env", default="HalfCheetah-v2")                  # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)                      # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--discount", default=0.99)                         # Discount factor
	parser.add_argument("--tau", default=0.005)                             # Target network update rate
	parser.add_argument("--random_head", action="store_true")               # Whether or not use random head
	parser.add_argument("--epsilon", default=0.05, type=float)              # random q head with epsilon
	parser.add_argument("--save_model", action="store_true")                # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                         # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print(f"---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print(f"---------------------------------------")
	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy.startswith("DDPG"):
		kwargs["random_head"] = args.random_head
		kwargs["epsilon"] = args.epsilon
		qhead_nums = 10
		policy = DDPG_aumc.DDPG_AUMC(**kwargs)
	elif args.policy.startswith("TD3"):
		kwargs["random_head"] = args.random_head
		kwargs["epsilon"] = args.epsilon
		qhead_nums = 10
		policy = TD3_aumc.TD3_AUMC(**kwargs)
	elif args.policy.startswith("SAC"):
		kwargs["random_head"] = args.random_head
		kwargs["epsilon"] = args.epsilon
		qhead_nums = 10
		policy = SAC_aumc.SAC_AUMC(**kwargs)
	else:
		raise ValueError(f"Don't support {args.policy}")

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		if not os.path.exists(f"./models/{policy_file}_actor"):
			raise ValueError(f"Not exist model `./models/{policy_file}_actor`!")
		policy.load(f"./models/{policy_file}")

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	
	for episode_num in range(10):
		state, done, ep_ret, ep_len = env.reset(), False, 0, 0
		while not done:
			if args.policy.startswith("SAC"):
				action = policy.select_action(np.array(state), True)
			else:
				action = policy.select_action(np.array(state))
			state, reward, done, _ = env.step(action)
			env.render()
			ep_ret += reward
			ep_len += 1
		
		if done:
			episode_num += 1
			print(f"Episode Num: {episode_num} Episode T: {ep_len} Reward: {ep_ret:.3f}")
