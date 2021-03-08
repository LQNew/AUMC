import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Soft Actor-Critic (SAC)
# Paper: https://arxiv.org/abs/1801.01290
# Soft Actor-Critic Algorithms and Applications
# https://arxiv.org/abs/1812.05905
# Implemetation of Attentive Update of Multi-Critic for Deep Reinforcement Learning (AUMC)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		# delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain("relu")
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def gaussian_logprob(noise, log_std):
	"""Compute Gaussian log probability."""
	residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
	return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

# Returns an action for a given state
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.mean = nn.Linear(hidden_dim, action_dim)
		self.log_std = nn.Linear(hidden_dim, action_dim)
		self.max_action = max_action
		self.apply(weight_init)
	
	def forward(self, state, deterministic=False, with_logprob=True):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mu_a = self.mean(a)
		log_std_a = self.log_std(a)
		log_std_a = torch.clamp(log_std_a, LOG_STD_MIN, LOG_STD_MAX)
		std_a = torch.exp(log_std_a)
		# Only used for evaluating policy at test time.
		if deterministic:
			z = mu_a
		else:
			noise = torch.randn_like(mu_a)  # sampled from guassian distribution
			z = mu_a + noise * std_a  # reparameterization trick
		action = torch.tanh(z) 

		if with_logprob and not deterministic:
			logp_pi = gaussian_logprob(noise, log_std_a).sum(axis=-1)
			logp_pi = logp_pi - (1.0 - action**2).clamp(min=1e-6).log().sum(axis=-1)
		else:
			logp_pi = None
		return self.max_action * action, logp_pi


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=256):
		super(Critic, self).__init__()
		# Q1 head ~ Q10 head
		self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

		self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l5 = nn.Linear(hidden_dim, hidden_dim)
		self.l6 = nn.Linear(hidden_dim, 1)

		self.l7 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l8 = nn.Linear(hidden_dim, hidden_dim)
		self.l9 = nn.Linear(hidden_dim, 1)

		self.l10 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l11 = nn.Linear(hidden_dim, hidden_dim)
		self.l12 = nn.Linear(hidden_dim, 1)

		self.l13 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l14 = nn.Linear(hidden_dim, hidden_dim)
		self.l15 = nn.Linear(hidden_dim, 1)

		self.l16 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l17 = nn.Linear(hidden_dim, hidden_dim)
		self.l18 = nn.Linear(hidden_dim, 1)

		self.l19 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l20 = nn.Linear(hidden_dim, hidden_dim)
		self.l21 = nn.Linear(hidden_dim, 1)
		
		self.l22 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l23 = nn.Linear(hidden_dim, hidden_dim)
		self.l24 = nn.Linear(hidden_dim, 1)

		self.l25 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l26 = nn.Linear(hidden_dim, hidden_dim)
		self.l27 = nn.Linear(hidden_dim, 1)

		self.l28 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l29 = nn.Linear(hidden_dim, hidden_dim)
		self.l30 = nn.Linear(hidden_dim, 1)
		self.apply(weight_init)

	def forward(self, state, action):
		sa = torch.cat([state, action], dim=1)
		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)

		q3 = F.relu(self.l7(sa))
		q3 = F.relu(self.l8(q3))
		q3 = self.l9(q3)

		q4 = F.relu(self.l10(sa))
		q4 = F.relu(self.l11(q4))
		q4 = self.l12(q4)

		q5 = F.relu(self.l13(sa))
		q5 = F.relu(self.l14(q5))
		q5 = self.l15(q5)

		q6 = F.relu(self.l16(sa))
		q6 = F.relu(self.l17(q6))
		q6 = self.l18(q6)

		q7 = F.relu(self.l19(sa))
		q7 = F.relu(self.l20(q7))
		q7 = self.l21(q7)

		q8 = F.relu(self.l22(sa))
		q8 = F.relu(self.l23(q8))
		q8 = self.l24(q8)

		q9 = F.relu(self.l25(sa))
		q9 = F.relu(self.l26(q9))
		q9 = self.l27(q9)

		q10 = F.relu(self.l28(sa))
		q10 = F.relu(self.l29(q10))
		q10 = self.l30(q10)
		return q1, q2, q3, q4, q5, q6, q7, q8, q9, q10

	def Qvalue(self, state, action, head=1):
		sa = torch.cat([state, action], dim=1)
		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)

		q3 = F.relu(self.l7(sa))
		q3 = F.relu(self.l8(q3))
		q3 = self.l9(q3)

		q4 = F.relu(self.l10(sa))
		q4 = F.relu(self.l11(q4))
		q4 = self.l12(q4)

		q5 = F.relu(self.l13(sa))
		q5 = F.relu(self.l14(q5))
		q5 = self.l15(q5)

		q6 = F.relu(self.l16(sa))
		q6 = F.relu(self.l17(q6))
		q6 = self.l18(q6)

		q7 = F.relu(self.l19(sa))
		q7 = F.relu(self.l20(q7))
		q7 = self.l21(q7)

		q8 = F.relu(self.l22(sa))
		q8 = F.relu(self.l23(q8))
		q8 = self.l24(q8)

		q9 = F.relu(self.l25(sa))
		q9 = F.relu(self.l26(q9))
		q9 = self.l27(q9)

		q10 = F.relu(self.l28(sa))
		q10 = F.relu(self.l29(q10))
		q10 = self.l30(q10)

		q_dict = {
			1: q1,
			2: q2,
			3: q3,
			4: q4,
			5: q5,
			6: q6,
			7: q7,
			8: q8,
			9: q9,
			10: q10
		}
		if head < 10: return q_dict[head], q_dict[head+1]
		else: return q_dict[10], q_dict[1]


class SAC_AUMC(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		random_head=False,
		double_qlearning=False,
		epsilon=0.01,
		hidden_dim=256,
	):
		self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
		self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
		self.target_entropy = -action_dim

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.random_head = random_head
		self.epsilon = epsilon

		self.head = 1
		self.random_head_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

	@property
	def alpha(self):
		return self.log_alpha.exp()

	def select_action(self, state, deterministic=False):
		state = torch.as_tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
		action, _ = self.actor(state, deterministic, False)
		return action.cpu().data.numpy().flatten()

	def td_error(self, state, action, reward, next_state, done):
		state = torch.as_tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
		next_state = torch.as_tensor(next_state.reshape(1, -1), device=device, dtype=torch.float32)
		action = torch.unsqueeze(torch.as_tensor(action, device=device, dtype=torch.float32), 0)
		next_action, logp_pi_next_action = self.actor(next_state)

		# Get current Q estimates
		Q1, Q2, Q3, Q4, Q5, Q6, \
			Q7, Q8, Q9, Q10 = self.critic(state, action)
		
		# Compute the target Q value
		target_Q1, target_Q2, target_Q3, target_Q4, target_Q5, target_Q6, \
			target_Q7, target_Q8, target_Q9, target_Q10 = self.critic_target(next_state, next_action)
		
		Q1  = Q1.cpu().data.numpy()[0][0]
		Q2  = Q2.cpu().data.numpy()[0][0]
		Q3  = Q3.cpu().data.numpy()[0][0]
		Q4  = Q4.cpu().data.numpy()[0][0]
		Q5  = Q5.cpu().data.numpy()[0][0]
		Q6  = Q6.cpu().data.numpy()[0][0]
		Q7  = Q7.cpu().data.numpy()[0][0]
		Q8  = Q8.cpu().data.numpy()[0][0]
		Q9  = Q9.cpu().data.numpy()[0][0]
		Q10 = Q10.cpu().data.numpy()[0][0]

		target_Q1  = target_Q1.cpu().data.numpy()[0][0]
		target_Q2  = target_Q2.cpu().data.numpy()[0][0]
		target_Q3  = target_Q3.cpu().data.numpy()[0][0]
		target_Q4  = target_Q4.cpu().data.numpy()[0][0]
		target_Q5  = target_Q5.cpu().data.numpy()[0][0]
		target_Q6  = target_Q6.cpu().data.numpy()[0][0]
		target_Q7  = target_Q7.cpu().data.numpy()[0][0]
		target_Q8  = target_Q8.cpu().data.numpy()[0][0]
		target_Q9  = target_Q9.cpu().data.numpy()[0][0]
		target_Q10 = target_Q10.cpu().data.numpy()[0][0]

		logp_pi_next_action = logp_pi_next_action.cpu().data.numpy()[0]

		td_error = []
		td_error.append(abs(reward + (1.0 - done) * self.discount * (target_Q1  - self.alpha.cpu().data.numpy() * logp_pi_next_action) - Q1))
		td_error.append(abs(reward + (1.0 - done) * self.discount * (target_Q2  - self.alpha.cpu().data.numpy() * logp_pi_next_action) - Q2))
		td_error.append(abs(reward + (1.0 - done) * self.discount * (target_Q3  - self.alpha.cpu().data.numpy() * logp_pi_next_action) - Q3))
		td_error.append(abs(reward + (1.0 - done) * self.discount * (target_Q4  - self.alpha.cpu().data.numpy() * logp_pi_next_action) - Q4))
		td_error.append(abs(reward + (1.0 - done) * self.discount * (target_Q5  - self.alpha.cpu().data.numpy() * logp_pi_next_action) - Q5))
		td_error.append(abs(reward + (1.0 - done) * self.discount * (target_Q6  - self.alpha.cpu().data.numpy() * logp_pi_next_action) - Q6))
		td_error.append(abs(reward + (1.0 - done) * self.discount * (target_Q7  - self.alpha.cpu().data.numpy() * logp_pi_next_action) - Q7))
		td_error.append(abs(reward + (1.0 - done) * self.discount * (target_Q8  - self.alpha.cpu().data.numpy() * logp_pi_next_action) - Q8))
		td_error.append(abs(reward + (1.0 - done) * self.discount * (target_Q9  - self.alpha.cpu().data.numpy() * logp_pi_next_action) - Q9))
		td_error.append(abs(reward + (1.0 - done) * self.discount * (target_Q10 - self.alpha.cpu().data.numpy() * logp_pi_next_action) - Q10))

		return np.array(td_error)

	def train(self, replay_buffer, batch_size=100, G=1):
		# Sample batches from replay buffer 
		state, action, next_state, reward, not_done, mask = replay_buffer.sample(batch_size)

		def random_generate():
			from random import shuffle
			l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
			shuffle(l)
			return l

		for _ in range(G):
			with torch.no_grad():
				# Select action according to policy 
				next_action, logp_pi_next_action = self.actor(next_state)
				logp_pi_next_action = torch.unsqueeze(logp_pi_next_action, 1)
				# Compute the target Q value
				target_Q1, target_Q2, target_Q3, target_Q4, target_Q5, target_Q6, \
					target_Q7, target_Q8, target_Q9, target_Q10 = self.critic_target(next_state, next_action)

				targetQ_dict = {
					1: target_Q1,
					2: target_Q2,
					3: target_Q3,
					4: target_Q4,
					5: target_Q5,
					6: target_Q6,
					7: target_Q7,
					8: target_Q8,
					9: target_Q9,
					10: target_Q10
				}
				if np.random.random_sample() < 0.005:
					self.random_head_list = random_generate()
				targetQ_dict[self.random_head_list[0]] = torch.min(targetQ_dict[self.random_head_list[0]], targetQ_dict[self.random_head_list[1]])
				targetQ_dict[self.random_head_list[1]] = targetQ_dict[self.random_head_list[0]]
				targetQ_dict[self.random_head_list[2]] = torch.min(targetQ_dict[self.random_head_list[2]], targetQ_dict[self.random_head_list[3]])
				targetQ_dict[self.random_head_list[3]] = targetQ_dict[self.random_head_list[2]] 
				targetQ_dict[self.random_head_list[4]] = torch.min(targetQ_dict[self.random_head_list[4]], targetQ_dict[self.random_head_list[5]])
				targetQ_dict[self.random_head_list[5]] = targetQ_dict[self.random_head_list[4]] 
				targetQ_dict[self.random_head_list[6]] = torch.min(targetQ_dict[self.random_head_list[6]], targetQ_dict[self.random_head_list[7]])
				targetQ_dict[self.random_head_list[7]] = targetQ_dict[self.random_head_list[6]] 
				targetQ_dict[self.random_head_list[8]] = torch.min(targetQ_dict[self.random_head_list[8]], targetQ_dict[self.random_head_list[9]])
				targetQ_dict[self.random_head_list[9]] = targetQ_dict[self.random_head_list[8]]
		
				target_Q1  = reward + not_done * self.discount * (target_Q1  - self.alpha.detach() * logp_pi_next_action)
				target_Q2  = reward + not_done * self.discount * (target_Q2  - self.alpha.detach() * logp_pi_next_action)
				target_Q3  = reward + not_done * self.discount * (target_Q3  - self.alpha.detach() * logp_pi_next_action)
				target_Q4  = reward + not_done * self.discount * (target_Q4  - self.alpha.detach() * logp_pi_next_action)
				target_Q5  = reward + not_done * self.discount * (target_Q5  - self.alpha.detach() * logp_pi_next_action)
				target_Q6  = reward + not_done * self.discount * (target_Q6  - self.alpha.detach() * logp_pi_next_action)
				target_Q7  = reward + not_done * self.discount * (target_Q7  - self.alpha.detach() * logp_pi_next_action)
				target_Q8  = reward + not_done * self.discount * (target_Q8  - self.alpha.detach() * logp_pi_next_action)
				target_Q9  = reward + not_done * self.discount * (target_Q9  - self.alpha.detach() * logp_pi_next_action)
				target_Q10 = reward + not_done * self.discount * (target_Q10 - self.alpha.detach() * logp_pi_next_action)
				
			# Get current Q estimates
			current_Q1, current_Q2, current_Q3, current_Q4, current_Q5, current_Q6, \
				current_Q7, current_Q8, current_Q9, current_Q10 = self.critic(state, action)
			
			target_Q1  = torch.unsqueeze(mask[:, 0], 1) * target_Q1  + (1 - torch.unsqueeze(mask[:, 0], 1)) * current_Q1
			target_Q2  = torch.unsqueeze(mask[:, 1], 1) * target_Q2  + (1 - torch.unsqueeze(mask[:, 1], 1)) * current_Q2
			target_Q3  = torch.unsqueeze(mask[:, 2], 1) * target_Q3  + (1 - torch.unsqueeze(mask[:, 2], 1)) * current_Q3
			target_Q4  = torch.unsqueeze(mask[:, 3], 1) * target_Q4  + (1 - torch.unsqueeze(mask[:, 3], 1)) * current_Q4
			target_Q5  = torch.unsqueeze(mask[:, 4], 1) * target_Q5  + (1 - torch.unsqueeze(mask[:, 4], 1)) * current_Q5
			target_Q6  = torch.unsqueeze(mask[:, 5], 1) * target_Q6  + (1 - torch.unsqueeze(mask[:, 5], 1)) * current_Q6
			target_Q7  = torch.unsqueeze(mask[:, 6], 1) * target_Q7  + (1 - torch.unsqueeze(mask[:, 6], 1)) * current_Q7
			target_Q8  = torch.unsqueeze(mask[:, 7], 1) * target_Q8  + (1 - torch.unsqueeze(mask[:, 7], 1)) * current_Q8
			target_Q9  = torch.unsqueeze(mask[:, 8], 1) * target_Q9  + (1 - torch.unsqueeze(mask[:, 8], 1)) * current_Q9
			target_Q10 = torch.unsqueeze(mask[:, 9], 1) * target_Q10 + (1 - torch.unsqueeze(mask[:, 9], 1)) * current_Q10

			# Compute critic loss
			critic_loss = 0.1 * (F.mse_loss(current_Q1, target_Q1) + F.mse_loss(current_Q2, target_Q2) + \
							F.mse_loss(current_Q3, target_Q3) + F.mse_loss(current_Q4, target_Q4) + \
							F.mse_loss(current_Q5, target_Q5) + F.mse_loss(current_Q6, target_Q6) + \
							F.mse_loss(current_Q7, target_Q7) + F.mse_loss(current_Q8, target_Q8) + \
							F.mse_loss(current_Q9, target_Q9) + F.mse_loss(current_Q10, target_Q10))

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Update the frozen target critic models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		# Compute actor loss
		action, logp_pi_action = self.actor(state)
		logp_pi_action = torch.unsqueeze(logp_pi_action, 1)
		if self.random_head:
			self.head = np.random.randint(1, 11)
			Q1_pi, Q2_pi = self.critic.Qvalue(state, action, self.head)
			Q_pi = torch.min(Q1_pi, Q2_pi)
		else:
			if np.random.random_sample() < self.epsilon:
				self.head = np.random.randint(1, 11)
			Q1_pi, Q2_pi = self.critic.Qvalue(state, action, self.head)
			Q_pi = torch.min(Q1_pi, Q2_pi)
		actor_loss = (self.alpha.detach() * logp_pi_action - Q_pi).mean()

		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()
	
		# Alpha loss, optimize the alpha
		self.alpha_optimizer.zero_grad()
		alpha_loss = -1.0 * (self.alpha * (logp_pi_action + self.target_entropy).detach()).mean()
		alpha_loss.backward()
		self.alpha_optimizer.step()

	# save the model
	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

	# load the model
	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))