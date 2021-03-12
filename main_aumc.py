import numpy as np
import random
import torch
import gym
import argparse
import os
import time

from DDPG import DDPG_aumc
from TD3 import TD3_aumc
from SAC import SAC_aumc
from utils import replay_buffer

from spinupUtils.logx import EpochLogger
from spinupUtils.run_utils import setup_logger_kwargs

def test_agent(policy, eval_env, seed, logger, eval_episodes=10):
	for _ in range(eval_episodes):
		state, done, ep_ret, ep_len = eval_env.reset(), False, 0, 0
		while not done:
			if args.policy.startswith("SAC"):
				action = policy.select_action(np.array(state), deterministic=True)
			else:
				action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			ep_ret += reward
			ep_len += 1
		logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="DDPG_aumc", type=str)          # Policy name
	parser.add_argument("--env", default="HalfCheetah-v2")                  # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)                      # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)        # Time steps initial random policy is used
	parser.add_argument("--start_timesteps_aumc", default=2e5, type=int)    # Time steps initial aumc masked samples generation is used
	parser.add_argument("--eval_freq", default=5e3, type=int)               # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)           # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                        # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)              # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                         # Discount factor
	parser.add_argument("--tau", default=0.005)                             # Target network update rate
	parser.add_argument("--mode", default="exp", type=str)                  # TD-errors for prob style
	parser.add_argument("--beta", default=0.4, type=float)                  # constant adding item 
	parser.add_argument("--random_head", action="store_true")               # Whether or not use random head
	parser.add_argument("--epsilon", default=0.05, type=float)              # random q head with epsilon
	parser.add_argument("--save_model", action="store_true")                # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                         # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--exp_name", type=str)       				        # Name for algorithms
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print(f"---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print(f"---------------------------------------")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, datestamp=False)
	logger = EpochLogger(**logger_kwargs)

	env = gym.make(args.env)
	eval_env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	eval_env.seed(args.seed)  # eval env for evaluating the agent
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
		policy.load(f"./models/{policy_file}")

	_replay_buffer = replay_buffer.BootstrappedReplayBuffer(state_dim, action_dim, qhead_nums=qhead_nums)
	
	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	start_time = time.time()
	mask = np.zeros(qhead_nums)

	for t in range(int(args.max_timesteps)):
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			if args.policy.startswith("SAC"):
				action = policy.select_action(np.array(state))
			else:
				action = (
					policy.select_action(np.array(state))
					+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
				).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		# If env stops when reaching max-timesteps, then `done_bool = False`, else `done_bool = True`
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
		if args.policy.endswith("aumc"):
			if t >= args.start_timesteps_aumc:
				td_errors = policy.td_error(state, action, reward, next_state, done_bool)
				if args.mode == "linear":
					td_errors_prob = td_errors / td_errors.sum()
					linear_prob = args.beta + td_errors_prob
					linear_prob = np.clip(linear_prob, 0.0, 1.0)
					for i in range(qhead_nums): mask[i] = np.random.binomial(1, linear_prob[i], 1)
				elif args.mode == "exp":
					td_error_max = np.max(td_errors) # trick for avoiding overflowing
					td_errors -= td_error_max
					# td_errors = np.clip(td_errors, 0.0, 705.0)  # in case for overflowing
					td_errors = np.exp(td_errors)
					td_errors_prob = td_errors / td_errors.sum()
					exp_prob = args.beta + td_errors_prob
					exp_prob = np.clip(exp_prob, 0.0, 1.0)
					for i in range(qhead_nums): mask[i] = np.random.binomial(1, exp_prob[i], 1)
				else:
					raise ValueError(f"Don't support {args.mode}!")
			else:
				mask = np.random.binomial(1, 1.0, qhead_nums)
		elif args.policy.endswith("bootstrapped"):
			mask = np.random.binomial(1, args.beta, qhead_nums)
		else:
			raise ValueError(f"Don't support {args.policy}!")

		# Store data in replay buffer
		_replay_buffer.add(state, action, next_state, reward, done_bool, mask)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(_replay_buffer, args.batch_size)

		if done: 
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			logger.store(EpRet=episode_reward, EpLen=episode_timesteps)
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		if (t + 1) % args.eval_freq == 0:
			test_agent(policy, eval_env, args.seed, logger)
			if args.save_model:
				policy.save(f"./models/{file_name}")
			logger.log_tabular("EpRet", with_min_and_max=True)
			logger.log_tabular("TestEpRet", with_min_and_max=True)
			logger.log_tabular("EpLen", average_only=True)
			logger.log_tabular("TestEpLen", average_only=True)
			logger.log_tabular("TotalEnvInteracts", t+1)
			logger.log_tabular("Time", time.time()-start_time)
			logger.dump_tabular()
