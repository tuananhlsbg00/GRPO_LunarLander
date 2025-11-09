# Scalable PyTorch RL Trainers (PPO & GRPO)

This repository provides high-performance, multiprocessing-based implementations of two key policy-gradient algorithms in PyTorch:

1.  **`PPOTrainer`**: A standard Actor-Critic **Proximal Policy Optimization (PPO)** trainer that is faithful to the design of Stable-Baselines3 (SB3). It uses **Generalized Advantage Estimation (GAE)** and is ideal for standard single-agent RL problems.
2.  **`GRPOTrainer`**: A flexible, Actor-Only **Group Reward Policy Optimization (GRPO)** trainer inspired by recent advancements in large-scale model optimization (like the [DeepSeek-RL paper](https://arxiv.org/abs/2405.04517)). This trainer is designed for experiments with batch-level advantage calculations and comes in three variations.

The core of this repository is a fast, `fork`-safe multiprocessing system that farms out rollouts to CPU workers, keeping the main process free for GPU-based optimization.

## 🚀 Features

* **High Performance**: Uses `multiprocessing.Pool` with a `fork` start method for efficient, low-overhead parallel data collection.
* **PPO (Actor-Critic)**: A `PPOTrainer` that uses a separate policy and value network, calculating advantages with GAE.
* **GRPO (Actor-Only)**: A flexible `GRPOBaseTrainer` with three variations:
    * **`PerStepAdvGRPOTrainer`**: (Process Supervision) Normalizes rewards at *each timestep* across all workers.
    * **`PerEpAdvGRPOTrainer`**: (Outcome Supervision) Normalizes the *total episode reward* and assigns this single advantage value to all steps in the episode.
    * **`HybridAdvGRPOTrainer`**: A custom implementation that combines Process and Outcome advantages using a "double-negative-proof" sign logic and multiplicative magnitude.
* **Configurable Seeding**: Use the `identical_G` config flag to toggle between all workers starting from the *same state* (for Process Supervision) or *different random states* (for Outcome/Hybrid Supervision).
* **Custom Environments**: Includes `DenseLunarLander` (with spawn randomization) and `SparseLunarLander` for robust testing.
* **Logging & Evaluation**:
    * Automatic TensorBoard logging to the `./runs` directory.
    * Built-in periodic policy evaluation.
    * Automatic video recording of your final, trained agent.

## 📦 Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  Install the required packages. It's highly recommended to use a virtual environment.
    ```bash
    pip install torch numpy gymnasium gymnasium[box2d] tensorboard tqdm
    ```

## 🤖 Available Algorithms

### 1. PPOTrainer (Actor-Critic)

This is a standard implementation of PPO-Clip, designed for general RL tasks.

* **Core Logic**: Actor-Critic. It trains a policy (`pi`) to select actions and a separate value network (`v`) to estimate the "goodness" of states.
* **Data Collection**: Gathers `G` trajectories of `T` steps each from parallel workers. These workers run with *different random seeds* to ensure high-variance, diverse data, which is standard for PPO.
* **Advantage**: Uses **Generalized Advantage Estimation (GAE)**, which combines `T` steps of rewards with the value function's estimate to get a low-variance advantage signal.

### 2. GRPOTrainer (Actor-Only)

This is a family of trainers (inheriting from `GRPOBaseTrainer`) that do **not** use a value function. They rely on normalizing rewards *across a batch* of `G` episodes to estimate the advantage. This is controlled by three distinct classes:

#### `PerEpAdvGRPOTrainer` (Outcome Supervision)
* **Core Logic**: Calculates the total reward (return) for each of the `G` episodes. It then standard-normalizes this vector of `G` returns.
* **Advantage**: Every single timestep `t` in an episode `i` receives the *exact same* advantage value: the normalized total return of episode `i`.
* **When to use**: When you only care about the final outcome of an episode and want to credit all actions equally.
* **`identical_G`**: Should be set to `False` (default) to get a diverse batch of outcomes.

#### `PerStepAdvGRPOTrainer` (Process Supervision)
* **Core Logic**: At each timestep `t`, it gathers the rewards $r_t$ from all `G` workers. It normalizes these `G` rewards *against each other*.
* **Advantage**: The advantage for a step `t` is the discounted sum of all *future normalized rewards* from that step onward.
* **When to use**: When you want to compare the "goodness" of different actions from the *exact same state*.
* **`identical_G`**: Should be set to `True` for this method to be statistically valid (as it assumes all workers start from the same state `s_0`).

#### `HybridAdvGRPOTrainer` (Custom Hybrid)
* **Core Logic**: A custom method that combines the two approaches. It calculates both the Episodic Advantage (`A_ep`) and the Step Advantage (`A_step`).
* **Advantage**: The hybrid advantage $A_{hybrid}$ is calculated with a special sign rule:
    * $sign = +1$ **if and only if** $A_{ep} > 0$ **and** $A_{step} > 0$.
    * $sign = -1$ otherwise.
    * The magnitude is multiplicative: $mag = |A_{ep}|^\alpha \times |A_{step}|^\beta$.
    * $A_{hybrid} = sign \times mag$
* **When to use**: When you want to credit good steps (`A_step > 0`) but only if they occurred in an overall good episode (`A_ep > 0`), solving the "double-negative" problem.
* **`identical_G`**: Should be set to `False` (default) to get diverse data for both advantage signals.

---

## ⚙️ Configuration & Hyperparameters

All trainers are configured using a `@dataclass`.

### PPOConfig (for `PPOTrainer`)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `env` | `Union[str, gym.Env]` | `"LunarLander-v3"` | Environment ID or a pre-initialized `gym.Env` object. |
| `gamma` | `float` | `0.99` | Discount factor for future rewards. |
| `lam` | `float` | `0.95` | The $\lambda$ parameter for Generalized Advantage Estimation (GAE). |
| `lr` | `float` | `3e-4` | Learning rate for the Adam optimizer (applied to both policy and value networks). |
| `clip_eps` | `float` | `0.2` | The clipping range $\epsilon$ for the PPO surrogate objective. |
| `clip_range_vf` | `Optional[float]` | `None` | Optional value function clipping (a la SB3). `None` disables it. |
| `vf_coef` | `float` | `0.5` | Coefficient for the value function loss in the total loss. |
| `ent_coef` | `float` | `0.01` | Coefficient for the entropy bonus (encourages exploration). |
| `max_grad_norm` | `float` | `0.5` | Maximum norm for gradient clipping. |
| `G` | `int` | `32` | Number of parallel workers to run. |
| `T` | `int` | `1024` | Number of steps *each worker* collects per iteration. **Total batch size = G * T**. |
| `epochs` | `int` | `10` | Number of optimization epochs to run on the collected batch. |
| `minibatches` | `int` | `32` | Number of minibatches to split the batch into during an epoch. |
| `n_workers` | `Optional[int]` | `None` | Number of multiprocessing pool workers. Defaults to `min(G, os.cpu_count())`. |
| `target_kl` | `Optional[float]` | `0.03` | If set, training will early-stop in an epoch if the KL-divergence exceeds this. |
| `normalize_advantage` | `bool` | `True` | Whether to normalize the GAE advantages before the update (standard practice). |
| `log_dir` | `str` | `"./runs/PPO"` | Base directory for TensorBoard logs. |
| `seed` | `Optional[int]` | `None` | Master seed for the run. |
| `verbose` | `int` | `1` | Whether to print training info to the console. |

### GRPOConfig (for all `GRPOTrainer` classes)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `env` | `Union[str, gym.Env]` | `"LunarLander-v3"` | Environment ID or a pre-initialized `gym.Env` object. |
| **`identical_G`** | **`bool`** | **`False`** | **If `True`, all `G` workers get the same seed. If `False`, all get different seeds.** |
| `gamma` | `float` | `0.99` | Discount factor (used in Process and Hybrid advantage calculation). |
| `lr` | `float` | `3e-4` | Learning rate for the Adam optimizer (applied to the *policy network only*). |
| `clip_eps` | `float` | `0.2` | The clipping range $\epsilon$ for the PPO surrogate objective. |
| `ent_coef` | `float` | `0.01` | Coefficient for the entropy bonus. |
| `max_grad_norm` | `float` | `0.5` | Maximum norm for gradient clipping. |
| `G` | `int` | `32` | Number of parallel episodes to run per iteration. |
| `T` | `int` | `1024` | **Max steps** per episode. Unlike PPO, total batch size is variable. |
| `epochs` | `int` | `8` | Number of optimization epochs to run on the collected batch. |
| `minibatches` | `int` | `16` | Number of minibatches to split the batch into during an epoch. |
| `n_workers` | `Optional[int]` | `None` | Number of multiprocessing pool workers. Defaults to `min(G, os.cpu_count())`. |
| `target_kl` | `float` | `0.015` | Target KL divergence for the adaptive $\beta_{KL}$ coefficient. |
| `beta_kl` | `float` | `0.02` | Initial value for the adaptive KL coefficient. |
| `kl_adjust_up` | `float` | `1.5` | Multiplier for $\beta_{KL}$ if `KL > target_kl`. |
| `kl_adjust_down` | `float` | `1 / 1.5` | Multiplier for $\beta_{KL}$ if `KL < target_kl`. |
| `ref_update_freq` | `int` | `1` | How many iterations before updating the reference policy `pi_ref`. |
| `hybrid_alpha` | `float` | `1.0` | Exponent $\alpha$ for the episodic term in `HybridAdvGRPOTrainer`. |
| `hybrid_beta` | `float` | `1.0` | Exponent $\beta$ for the step term in `HybridAdvGRPOTrainer`. |
| `hybrid_clip` | `Optional[float]` | `10.0` | Clips the hybrid advantage magnitude before final normalization. |
| `log_dir` | `str` | `"./runs/GRPO"` | Base directory for TensorBoard logs. |
| `seed` | `Optional[int]` | `None` | Master seed for the run. |
| `verbose` | `int` | `1` | Whether to print training info to the console. |

---

## 🏃 How to Run (Example Scripts)

You will need to have your policy classes (e.g., `ActorCriticPolicy`, `SimpleGRPOPolicy`) and custom environments (`DenseLunarLander`) available in your `dense_scripts.utils` module.

### 1. Standard PPO (Actor-Critic)

This example uses the `PPOTrainer` with GAE. It's the standard, robust choice for `LunarLander`.

```python
import torch
# PPO needs an Actor-Critic policy (or any policy with .dist() and .value() methods)
from dense_scripts.utils.policies import ActorCriticPolicy 
from dense_scripts.PPO.ppo import PPOTrainer, PPOConfig
from dense_scripts.utils.envs import DenseLunarLander

# 1. Initialize environments
env_train = DenseLunarLander(randomize_angle=True, randomize_pos=True)
env_eval = DenseLunarLander(randomize_angle=False, randomize_pos=False)

# 2. Initialize the Policy
# The PPOTrainer will automatically create its own ValueNet, 
# but it's cleaner to use a combined ActorCriticPolicy.
policy = ActorCriticPolicy(env_train.observation_space.shape[0], env_train.action_space.n)

# 3. Configure the Trainer
cfg = PPOConfig(
    env=env_train,
    G=32,
    T=1024,
    gamma=0.99,
    lam=0.95,
    log_dir="./runs/PPO_Lander"
)

# 4. Initialize the Trainer
trainer = PPOTrainer(policy, cfg, device="cpu") # or "cuda"

# 5. Run Training
trainer.train(
    iters=300,
    eval_env=env_eval,
    eval_interval=30,
    eval_episodes=100,
    video_dir="videos/PPO_Lander",
    video_episodes=10
)
```


### 2. GRPO - Process Supervision (Actor-Only)
This example uses `PerStepAdvGRPOTrainer`. Note that we should set `identical_G=True`.


```python   
import torch
from dense_scripts.utils.policies import SimpleGRPOPolicy # GRPO is Actor-Only
from dense_scripts.GRPO.grpo import PerStepAdvGRPOTrainer, GRPOConfig
from dense_scripts.utils.envs import DenseLunarLander

# 1. Initialize environments
# We use a non-randomized env for the process supervision task
env_train = DenseLunarLander(randomize_angle=False, randomize_pos=False)
env_eval = DenseLunarLander(randomize_angle=False, randomize_pos=False)

# 2. Initialize the Policy
policy = SimpleGRPOPolicy(env_train.observation_space.shape[0], env_train.action_space.n)

# 3. Configure the Trainer
cfg = GRPOConfig(
    env=env_train, 
    G=32, 
    T=1024, 
    gamma=0.99, 
    log_dir="./runs/GRPO_Process",
    identical_G=True  # <-- CRITICAL: All workers must start from the same seed
)

# 4. Initialize the Trainer
trainer = PerStepAdvGRPOTrainer(policy, cfg, device="cpu")

# 5. Run Training
trainer.train(
    iters=300,
    eval_env=env_eval,
    eval_interval=30,
    eval_episodes=100,
    video_dir="videos/GRPO_Process",
    video_episodes=10
)
```


### 3. GRPO - Outcome Supervision (Actor-Only)
This example uses `PerEpAdvGRPOTrainer`. We should use `identical_G=False` to get diverse rollouts.


```python   

import torch
from dense_scripts.utils.policies import SimpleGRPOPolicy
from dense_scripts.GRPO.grpo import PerEpAdvGRPOTrainer, GRPOConfig
from dense_scripts.utils.envs import DenseLunarLander

# 1. Initialize environments (randomized is good for this mode)
env_train = DenseLunarLander(randomize_angle=True, randomize_pos=True)
env_eval = DenseLunarLander(randomize_angle=False, randomize_pos=False)

# 2. Initialize the Policy
policy = SimpleGRPOPolicy(env_train.observation_space.shape[0], env_train.action_space.n)

# 3. Configure the Trainer
cfg = GRPOConfig(
    env=env_train, 
    G=32, 
    T=1024, 
    gamma=0.99, 
    log_dir="./runs/GRPO_Outcome",
    identical_G=False # <-- We want diverse, random rollouts
)

# 4. Initialize the Trainer
trainer = PerEpAdvGRPOTrainer(policy, cfg, device="cpu")

# 5. Run Training
trainer.train(
    iters=300,
    eval_env=env_eval,
    eval_interval=30,
    eval_episodes=100,
    video_dir="videos/GRPO_Outcome",
    video_episodes=10
)
```


### 4. GRPO - Hybrid Advantage (Actor-Only)
This example uses `HybridAdvGRPOTrainer`. We also should use `identical_G=False`.


```python   

import torch
from dense_scripts.utils.policies import SimpleGRPOPolicy
from dense_scripts.GRPO.grpo import HybridAdvGRPOTrainer, GRPOConfig
from dense_scripts.utils.envs import DenseLunarLander

# 1. Initialize environments (randomized is good for this mode)
env_train = DenseLunarLander(randomize_angle=True, randomize_pos=True)
env_eval = DenseLunarLander(randomize_angle=False, randomize_pos=False)

# 2. Initialize the Policy
policy = SimpleGRPOPolicy(env_train.observation_space.shape[0], env_train.action_space.n)

# 3. Configure the Trainer
cfg = GRPOConfig(
    env=env_train, 
    G=32, 
    T=1024, 
    gamma=0.99, 
    log_dir="./runs/GRPO_Hybrid",
    identical_G=False, # <-- We want diverse, random rollouts
    
    # Tune the hybrid-specific parameters
    hybrid_alpha=1.0,
    hybrid_beta=1.0,
    hybrid_clip=10.0
)

# 4. Initialize the Trainer
trainer = HybridAdvGRPOTrainer(policy, cfg, device="cpu")

# 5. Run Training
trainer.train(
    iters=300,
    eval_env=env_eval,
    eval_interval=30,
    eval_episodes=100,
    video_dir="videos/GRPO_Hybrid",
    video_episodes=10
)
```
