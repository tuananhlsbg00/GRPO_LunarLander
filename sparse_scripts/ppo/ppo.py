"""
Pure PPO Implementation (PyTorch + Gymnasium)

This is a clean, minimal-yet-complete PPO implementation compatible with Gymnasium
environments. It mirrors the standard PPO algorithm:

Objective (per minibatch):
  L = E[min(r*A, clip(r,1-ε,1+ε)*A)]
	  + c1 * (V_target - V)^2
	  + c2 * H[π]

Features:
  - Actor-Critic network (policy + value)
  - Discrete and continuous action spaces
  - GAE(λ) advantage estimation
  - Clipped policy objective
  - Entropy bonus, value loss
  - Gradient clipping
  - Save/Load utilities

Note: This implementation is self-contained and does not depend on Stable Baselines3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Categorical, Normal
import gymnasium as gym


DEFAULT_PPO_CONFIG: Dict[str, Any] = {
	"learning_rate": 2e-4,
	"n_steps": 2048,
	"batch_size": 64,
	"n_epochs": 10,
	"gamma": 1.0,
	"gae_lambda": 0.95,
	"clip_range": 0.2,
	"ent_coef": 0.01,
	"vf_coef": 0.5,
	"max_grad_norm": 0.5,
	"beta": 0.04,
	"device": "cpu",
	"verbose": 0,
	"hidden_dims": [64, 64],
}


def default_ppo_config(**overrides: Any) -> Dict[str, Any]:
	"""Return a copy of the PPO defaults with optional overrides."""

	config = DEFAULT_PPO_CONFIG.copy()
	config.update(overrides)
	if "hidden_dims" in config and config["hidden_dims"] is not None:
		config["hidden_dims"] = list(config["hidden_dims"])
	return config


# =============================================================
# Networks
# =============================================================

class ActorCritic(nn.Module):
	"""
	Actor-Critic network with a shared MLP backbone.

	- For discrete actions: policy outputs logits.
	- For continuous actions: policy outputs mean and (learned) log_std.
	- Value head outputs scalar V(s).
	"""

	def __init__(
		self,
		obs_dim: int,
		action_dim: int,
		hidden_dims: List[int] = [64, 64],
		action_space_type: str = "discrete",
	) -> None:
		super().__init__()
		self.action_space_type = action_space_type

		layers: List[nn.Module] = []
		last_dim = obs_dim
		for h in hidden_dims:
			layers += [nn.Linear(last_dim, h), nn.Tanh()]
			last_dim = h
		self.backbone = nn.Sequential(*layers)

		# Policy heads
		if action_space_type == "discrete":
			self.pi = nn.Linear(last_dim, action_dim)
			self.log_std_param = None
		elif action_space_type == "continuous":
			self.mu = nn.Linear(last_dim, action_dim)
			self.log_std_param = nn.Parameter(th.zeros(action_dim))
		else:
			raise ValueError(f"Unsupported action space type: {action_space_type}")

		# Value head
		self.v = nn.Linear(last_dim, 1)

	def forward(self, obs: th.Tensor) -> Dict[str, th.Tensor]:
		h = self.backbone(obs)
		if self.action_space_type == "discrete":
			logits = self.pi(h)
			return {"logits": logits, "value": self.v(h).squeeze(-1)}
		else:
			mu = self.mu(h)
			log_std = self.log_std_param.expand_as(mu)
			return {"mu": mu, "log_std": log_std, "value": self.v(h).squeeze(-1)}

	def dist(self, obs: th.Tensor):
		out = self.forward(obs)
		if self.action_space_type == "discrete":
			return Categorical(logits=out["logits"]), out["value"]
		else:
			std = th.exp(out["log_std"])  # type: ignore[index]
			return Normal(out["mu"], std), out["value"]  # type: ignore[index]


# =============================================================
# Rollout buffer with GAE(λ)
# =============================================================

@dataclass
class Transition:
	obs: np.ndarray
	action: np.ndarray
	reward: float
	done: bool
	value: float
	log_prob: float


class RolloutBuffer:
	def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...], device: str) -> None:
		self.capacity = capacity
		self.device = device
		self.reset(obs_shape, action_shape)

	def reset(self, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...]):
		self.pos = 0
		self.full = False
		self.observations = np.zeros((self.capacity,) + obs_shape, dtype=np.float32)
		self.actions = np.zeros((self.capacity,) + action_shape, dtype=np.float32)
		self.rewards = np.zeros((self.capacity,), dtype=np.float32)
		self.dones = np.zeros((self.capacity,), dtype=np.float32)
		self.values = np.zeros((self.capacity,), dtype=np.float32)
		self.log_probs = np.zeros((self.capacity,), dtype=np.float32)
		self.advantages = np.zeros((self.capacity,), dtype=np.float32)
		self.returns = np.zeros((self.capacity,), dtype=np.float32)

	def add(self, tr: Transition):
		self.observations[self.pos] = tr.obs
		self.actions[self.pos] = tr.action
		self.rewards[self.pos] = tr.reward
		self.dones[self.pos] = float(tr.done)
		self.values[self.pos] = tr.value
		self.log_probs[self.pos] = tr.log_prob
		self.pos += 1
		if self.pos >= self.capacity:
			self.full = True

	def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
		adv = 0.0
		for t in reversed(range(self.capacity)):
			mask = 1.0 - self.dones[t]
			next_value = last_value if t == self.capacity - 1 else self.values[t + 1]
			delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
			adv = delta + gamma * gae_lambda * mask * adv
			self.advantages[t] = adv
			self.returns[t] = self.advantages[t] + self.values[t]

	def get(self, batch_size: int):
		# Normalize advantages globally
		adv = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

		idxs = np.random.permutation(self.capacity)
		for start in range(0, self.capacity, batch_size):
			end = start + batch_size
			mb_idx = idxs[start:end]
			yield (
				th.tensor(self.observations[mb_idx], dtype=th.float32, device=self.device),
				th.tensor(self.actions[mb_idx], dtype=th.float32, device=self.device),
				th.tensor(self.log_probs[mb_idx], dtype=th.float32, device=self.device),
				th.tensor(self.returns[mb_idx], dtype=th.float32, device=self.device),
				th.tensor(adv[mb_idx], dtype=th.float32, device=self.device),
				th.tensor(self.values[mb_idx], dtype=th.float32, device=self.device),
			)


# =============================================================
# PPO algorithm
# =============================================================


class PPO:
	def __init__(
		self,
		env: gym.Env,
		policy_network: Optional[ActorCritic] = None,
		learning_rate: float = 3e-4,
		n_steps: int = 2048,
		batch_size: int = 64,
		n_epochs: int = 10,
		gamma: float = 0.99,
		gae_lambda: float = 0.95,
		clip_range: float = 0.2,
		ent_coef: float = 0.0,
		vf_coef: float = 0.5,
		max_grad_norm: float = 0.5,
		beta: float = 0.0,
		device: str = "cpu",
		verbose: int = 1,
	) -> None:
		self.env = env
		self.device = device
		self.lr = learning_rate
		self.n_steps = n_steps
		self.batch_size = batch_size
		self.n_epochs = n_epochs
		self.gamma = gamma
		self.gae_lambda = gae_lambda
		self.clip_range = clip_range
		self.ent_coef = ent_coef
		self.vf_coef = vf_coef
		self.max_grad_norm = max_grad_norm
		self.beta = beta
		self.verbose = verbose

		# Detect spaces
		if isinstance(env.action_space, gym.spaces.Discrete):
			self.action_space_type = "discrete"
			action_dim = env.action_space.n
			action_shape = (1,)
		elif isinstance(env.action_space, gym.spaces.Box):
			self.action_space_type = "continuous"
			action_dim = env.action_space.shape[0]
			action_shape = (action_dim,)
		else:
			raise ValueError(f"Unsupported action space: {env.action_space}")

		assert isinstance(env.observation_space, gym.spaces.Box), "Only Box observation space is supported"
		obs_dim = int(np.prod(env.observation_space.shape))
		obs_shape = (obs_dim,)

		# Policy
		self.policy = policy_network or ActorCritic(
			obs_dim=obs_dim,
			action_dim=action_dim,
			hidden_dims=[64, 64],
			action_space_type=self.action_space_type,
		)
		self.policy.to(self.device)

		# Optimizer
		self.optimizer = th.optim.Adam(self.policy.parameters(), lr=self.lr)

		# Rollout buffer
		self.buffer = RolloutBuffer(self.n_steps, obs_shape, action_shape, device=self.device)

		# Logging
		self.num_timesteps = 0
		self.episode_rewards: List[float] = []
		self.episode_lengths: List[int] = []

	# ---------------------------------------------------------
	# Interaction helpers
	# ---------------------------------------------------------
	def _obs_to_tensor(self, obs: np.ndarray) -> th.Tensor:
		obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
		return th.tensor(obs, dtype=th.float32, device=self.device)

	def _sample_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
		obs_t = self._obs_to_tensor(obs)
		dist, value = self.policy.dist(obs_t)
		if self.action_space_type == "discrete":
			action = dist.sample()
			log_prob = dist.log_prob(action)
			act_np = action.cpu().numpy().astype(np.int64)
			act_env = act_np[0]
		else:
			action = dist.sample()
			log_prob = dist.log_prob(action).sum(-1)
			act_env = action.cpu().numpy()[0]
		return act_env, float(log_prob.item()), float(value.item())

	# ---------------------------------------------------------
	# Rollout collection
	# ---------------------------------------------------------
	def _collect_rollout(self) -> Tuple[np.ndarray, float, bool, Dict]:
		# Reset buffer
		self.buffer.reset(self.buffer.observations.shape[1:], self.buffer.actions.shape[1:])

		# Reset env (Gymnasium API returns (obs, info))
		obs = self.env.reset()
		if isinstance(obs, tuple):
			obs, _ = obs

		ep_reward = 0.0
		ep_length = 0
		last_info: Dict = {}
		last_done = False
		last_value = 0.0

		for step in range(self.n_steps):
			action, log_prob, value = self._sample_action(obs)
			step_result = self.env.step(action)
			if len(step_result) == 5:
				next_obs, reward, terminated, truncated, info = step_result
			else:
				next_obs, reward, done, info = step_result
				terminated, truncated = done, False

			done_flag = bool(terminated or truncated)

			self.buffer.add(
				Transition(
					obs=np.asarray(obs, dtype=np.float32).reshape(-1),
					action=np.asarray(action, dtype=np.float32 if self.action_space_type == "continuous" else np.int64).reshape(-1),
					reward=float(reward),
					done=done_flag,
					value=value,
					log_prob=log_prob,
				)
			)

			self.num_timesteps += 1
			ep_reward += float(reward)
			ep_length += 1

			obs = next_obs
			last_info = info
			last_done = done_flag
			last_value = value if done_flag else 0.0  # placeholder; will replace below

			if done_flag:
				self.episode_rewards.append(ep_reward)
				self.episode_lengths.append(ep_length)
				# Reset episode
				ep_reward, ep_length = 0.0, 0
				reset_res = self.env.reset()
				if isinstance(reset_res, tuple):
					obs, _ = reset_res
				else:
					obs = reset_res

		# Bootstrap last value with current value function if not done
		obs_t = self._obs_to_tensor(obs)
		with th.no_grad():
			_, v_bootstrap = self.policy.dist(obs_t)
		last_value = float(v_bootstrap.item()) if not last_done else 0.0

		self.buffer.compute_gae(last_value, gamma=self.gamma, gae_lambda=self.gae_lambda)
		return obs, last_value, last_done, last_info

	# ---------------------------------------------------------
	# Update
	# ---------------------------------------------------------
	def _update(self):
		policy_losses = []
		value_losses = []
		entropy_losses = []
		kl_penalties = []
		approx_kls = []
		clip_fracs = []

		for _ in range(self.n_epochs):
			for obs_b, act_b, old_logp_b, ret_b, adv_b, val_b in self.buffer.get(self.batch_size):
				# Build distribution and value
				dist, value = self.policy.dist(obs_b)
				if self.action_space_type == "discrete":
					act_b_long = act_b.long().squeeze(-1)
					logp = dist.log_prob(act_b_long)
					entropy = dist.entropy()
				else:
					logp = dist.log_prob(act_b).sum(-1)
					entropy = dist.entropy().sum(-1)

				ratio = th.exp(logp - old_logp_b)
				# Policy loss
				unclipped = adv_b * ratio
				clipped = adv_b * th.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
				policy_loss = -th.min(unclipped, clipped).mean()

				# Value loss (MSE)
				value_loss = th.mean((ret_b - value) ** 2)

				# Entropy bonus (maximize entropy => minimize -entropy)
				entropy_loss = -entropy.mean()

				kl_term = th.mean(logp - old_logp_b)
				loss = (
					policy_loss
					+ self.vf_coef * value_loss
					+ self.ent_coef * entropy_loss
					+ self.beta * kl_term
				)

				self.optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
				self.optimizer.step()

				with th.no_grad():
					log_ratio = logp - old_logp_b
					approx_kl = th.mean((th.exp(log_ratio) - 1) - log_ratio)
					clip_frac = th.mean((th.abs(ratio - 1.0) > self.clip_range).float())
					approx_kls.append(float(approx_kl.item()))
					clip_fracs.append(float(clip_frac.item()))
					policy_losses.append(float(policy_loss.item()))
					value_losses.append(float(value_loss.item()))
					entropy_losses.append(float((-entropy_loss).item()))
					kl_penalties.append(float(kl_term.item()))

		stats = {
			"policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
			"value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
			"entropy": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
			"kl_penalty": float(np.mean(kl_penalties)) if kl_penalties else 0.0,
			"approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
			"clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
		}
		return stats

	# ---------------------------------------------------------
	# Public API
	# ---------------------------------------------------------
	def learn(self, total_timesteps: int, log_interval: int = 1):
		if self.verbose:
			print("=" * 70)
			print("PPO Training")
			print("=" * 70)
			print(f"n_steps: {self.n_steps}, batch_size: {self.batch_size}, epochs: {self.n_epochs}")
			print(f"gamma: {self.gamma}, gae_lambda: {self.gae_lambda}, clip_range: {self.clip_range}")
			print(f"ent_coef: {self.ent_coef}, vf_coef: {self.vf_coef}")
			print("=" * 70)

		iteration = 0
		while self.num_timesteps < total_timesteps:
			iteration += 1
			self._collect_rollout()
			stats = self._update()

			if self.verbose and (iteration % log_interval == 0):
				mean_r = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
				mean_l = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0.0
				print(f"Iter {iteration} | Steps {self.num_timesteps}/{total_timesteps}")
				print(f"  EpReward(100): {mean_r:.2f} | EpLen(100): {mean_l:.1f}")
				print(
					f"  policy_loss: {stats['policy_loss']:.4f} | value_loss: {stats['value_loss']:.4f} | "
					f"entropy: {stats['entropy']:.3f} | kl_penalty: {stats['kl_penalty']:.4f} | "
					f"kl: {stats['approx_kl']:.4f} | clip_frac: {stats['clip_frac']:.3f}"
				)

		if self.verbose:
			print("Training completed.")

	def evaluate(self, n_episodes: int = 5, deterministic: bool = True) -> Dict[str, float]:
		rewards = []
		lengths = []
		for _ in range(n_episodes):
			res = self.env.reset()
			obs, _ = res if isinstance(res, tuple) else (res, {})
			done = False
			ep_r = 0.0
			ep_l = 0
			while not done:
				obs_t = self._obs_to_tensor(obs)
				dist, _ = self.policy.dist(obs_t)
				if deterministic:
					if self.action_space_type == "discrete":
						action = th.argmax(dist.probs, dim=-1)  # type: ignore[attr-defined]
						act = int(action.item())
					else:
						act = dist.mean.cpu().numpy()[0]  # type: ignore[attr-defined]
				else:
					if self.action_space_type == "discrete":
						act = int(dist.sample().item())
					else:
						act = dist.sample().cpu().numpy()[0]

				step_res = self.env.step(act)
				if len(step_res) == 5:
					obs, r, term, trunc, _ = step_res
					done = bool(term or trunc)
				else:
					obs, r, done, _ = step_res
				ep_r += float(r)
				ep_l += 1
			rewards.append(ep_r)
			lengths.append(ep_l)
		return {
			"mean_reward": float(np.mean(rewards)) if rewards else 0.0,
			"mean_length": float(np.mean(lengths)) if lengths else 0.0,
		}

	def save(self, path: str):
		th.save({"state_dict": self.policy.state_dict()}, path)

	def load(self, path: str):
		data = th.load(path, map_location=self.device)
		self.policy.load_state_dict(data["state_dict"])


# =============================================================
# Factory
# =============================================================


def make_ppo_agent(env: gym.Env, **kwargs) -> PPO:
	config = default_ppo_config(**kwargs)
	policy_network = config.pop("policy_network", None)
	hidden_dims = config.pop("hidden_dims", [64, 64])
	if hidden_dims is None:
		hidden_dims = [64, 64]

	if policy_network is None:
		if isinstance(env.action_space, gym.spaces.Discrete):
			action_dim = env.action_space.n
			action_space_type = "discrete"
		elif isinstance(env.action_space, gym.spaces.Box):
			action_dim = env.action_space.shape[0]
			action_space_type = "continuous"
		else:
			raise ValueError(f"Unsupported action space: {env.action_space}")

		if not isinstance(env.observation_space, gym.spaces.Box):
			raise ValueError(f"Unsupported observation space: {env.observation_space}")

		obs_dim = int(np.prod(env.observation_space.shape))
		policy_network = ActorCritic(
			obs_dim=obs_dim,
			action_dim=action_dim,
			hidden_dims=hidden_dims,
			action_space_type=action_space_type,
		)

	return PPO(env=env, policy_network=policy_network, **config)


def load_trained_ppo(model_path: str, env_factory: Callable[[], gym.Env], **kwargs) -> PPO:
	"""Load a trained PPO agent using an environment factory."""

	env = env_factory()
	agent = make_ppo_agent(env, **kwargs)
	agent.load(model_path)
	agent.policy.eval()
	return agent


def evaluate_policy(
	agent: PPO,
	env: gym.Env,
	n_episodes: int = 10,
	deterministic: bool = False,
	render: bool = False,
) -> Dict[str, Any]:
	"""Run evaluation episodes and return aggregated metrics."""

	successes: List[float] = []
	rewards: List[float] = []
	lengths: List[int] = []
	crash_details: List[Dict[str, Any]] = []

	for _ in range(n_episodes):
		reset_res = env.reset()
		obs, info = reset_res if isinstance(reset_res, tuple) else (reset_res, {})
		done = False
		ep_reward = 0.0
		ep_length = 0
		last_info: Dict[str, Any] = info

		while not done:
			obs_tensor = agent._obs_to_tensor(obs)
			dist, _ = agent.policy.dist(obs_tensor)
			if deterministic:
				if agent.action_space_type == "discrete":
					action = th.argmax(dist.probs, dim=-1)  # type: ignore[attr-defined]
					act = int(action.item())
				else:
					act = dist.mean.cpu().numpy()[0]  # type: ignore[attr-defined]
			else:
				if agent.action_space_type == "discrete":
					act = int(dist.sample().item())
				else:
					act = dist.sample().cpu().numpy()[0]

			step_res = env.step(act)
			if len(step_res) == 5:
				obs, reward, terminated, truncated, info = step_res
				done = bool(terminated or truncated)
			else:
				obs, reward, done, info = step_res

			last_info = info
			ep_reward += float(reward)
			ep_length += 1

			if render:
				env.render()

		success = float(1.0 if last_info.get("landing_success", False) else 0.0)
		successes.append(success)
		rewards.append(ep_reward)
		lengths.append(ep_length)

		if not success:
			crash_details.append(
				{
					"velocity": last_info.get("crash_velocity", 0),
					"legs_touching": last_info.get("legs_touching", 0),
					"penalty": last_info.get("crash_penalty", 0),
				}
			)

	return {
		"success_rate": float(np.mean(successes)) if successes else 0.0,
		"success_std": float(np.std(successes)) if successes else 0.0,
		"mean_reward": float(np.mean(rewards)) if rewards else 0.0,
		"reward_std": float(np.std(rewards)) if rewards else 0.0,
		"mean_length": float(np.mean(lengths)) if lengths else 0.0,
		"length_std": float(np.std(lengths)) if lengths else 0.0,
		"episode_rewards": rewards,
		"episode_lengths": lengths,
		"successes": successes,
		"n_episodes": n_episodes,
		"crash_details": crash_details,
	}

