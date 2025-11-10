# <dense_script/utils/tools.py>
import torch
import gymnasium as gym
import numpy as np
from pathlib import Path
import imageio.v2 as imageio


from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple, List, Type, Union


def record_videos(
    policy,
    env="LunarLander-v3",
    video_dir="videos",
    episodes=3,
    fps=30,
    device="cuda",
    out_format="mp4",  # "mp4" or "gif"
    T=1024  # max steps per episode
):
    """
    Record policy rollouts as videos.

    Supports:
      - env: Gym ID string
      - env: environment instance
      - env: (class, kwargs) spec (as used in GRPO)
    """
    Path(video_dir).mkdir(parents=True, exist_ok=True)

    # --- Create or prepare the environment ---
    if isinstance(env, str):
        # If a string ID, build the env with RGB rendering
        env_instance = gym.make(env, render_mode="rgb_array")
        env_name = env
    elif hasattr(env, "reset"):
        # If it's already an environment instance
        env_instance = env
        env_name = getattr(env_instance, "spec", None)
        env_name = env_name.id if env_name else env_instance.__class__.__name__

        # ✅ Ensure render_mode is set for proper video rendering
        if not hasattr(env_instance, "render_mode") or env_instance.render_mode != "rgb_array":
            try:
                env_instance.render_mode = "rgb_array"
            except Exception:
                pass
    else:
        raise TypeError(
            f"Unsupported env type: {type(env)}. Expected str or gym.Env instance."
        )

    print(f"🎥 Recording {episodes} episodes from {env_name}...")

    for ep in range(episodes):
        frames = []
        obs, _ = env_instance.reset()
        done = truncated = False
        total_reward = 0.0
        steps = 0

        while not (done or truncated):
            if T and steps >= T:
                break
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if hasattr(policy, "policy"):
                    dist = policy.policy.dist(obs_t)
                else:
                    dist = policy.dist(obs_t)
                action = dist.probs.argmax(dim=-1).item()  # greedy
            obs, reward, done, truncated, _ = env_instance.step(action)
            total_reward += reward
            frame = env_instance.render()
            if frame is not None:
                frame = np.array(frame)
                if frame.dtype != np.uint8:
                    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                frames.append(frame)
            steps += 1

        # --- Save file ---
        filename = f"{env_name}_ep{ep+1:02d}_R{total_reward:.1f}.{out_format}"
        video_path = Path(video_dir) / filename
        if out_format.lower() == "mp4":
            imageio.mimsave(video_path, frames, fps=fps, codec="libx264", quality=8)
        elif out_format.lower() == "gif":
            imageio.mimsave(video_path, frames, fps=fps)
        else:
            raise ValueError("out_format must be 'mp4' or 'gif'")

        print(f"✅ Saved {out_format.upper()} video: {video_path} | Reward: {total_reward:.1f}")

    env_instance.close()


# ===========================================
# Utilities to carry env specs across processes
# ===========================================

EnvSpec = Union[str, Tuple[Type[gym.Env], Dict[str, Any]]]

def make_env_from_spec(spec: EnvSpec) -> gym.Env:
    if isinstance(spec, str):
        return gym.make(spec)
    cls, kwargs = spec
    return cls(**kwargs)

def env_to_spec(env_like: Union[str, gym.Env]) -> Tuple[EnvSpec, str]:
    """
    Convert an env id or env object into a picklable spec + a nice env_id string for logging.
    """
    if isinstance(env_like, str):
        return env_like, env_like
    # env object: we expect it to expose _init_kwargs (our DenseLunarLander does)
    kwargs = getattr(env_like, "_init_kwargs", None)
    if kwargs is None:
        # Fallback: try to reconstruct from common attributes
        kwargs = dict(
            render_mode=getattr(env_like, "render_mode", None),
            continuous=getattr(env_like, "continuous", False),
            gravity=getattr(env_like, "gravity", -10.0),
            enable_wind=getattr(env_like, "enable_wind", False),
            wind_power=getattr(env_like, "wind_power", 15.0),
            turbulence_power=getattr(env_like, "turbulence_power", 1.5),
        )
    cls = env_like.__class__
    env_id_str = getattr(getattr(env_like, "spec", None), "id", cls.__name__)
    return (cls, kwargs), env_id_str

import torch, numpy as np, gymnasium as gym
from multiprocessing import Pool, cpu_count, get_context
from tqdm import tqdm

def _evaluate_episode(args):
    model, env_spec, device, T = args

    # 1️⃣ Rebuild environment safely from spec
    if isinstance(env_spec, str):
        env = gym.make(env_spec)
    elif isinstance(env_spec, tuple):
        env = make_env_from_spec(env_spec)
    elif callable(env_spec):
        env = env_spec()
    else:
        raise TypeError(f"Unsupported env type for evaluation: {type(env_spec)}")

    obs, _ = env.reset()
    done = truncated = False
    ep_reward, steps = 0.0, 0
    info = {}

    while not (done or truncated):
        if T and steps >= T:
            break
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if hasattr(model, "policy"):
                dist = model.policy.dist(obs_t)
            else:
                dist = model.dist(obs_t)
            action = dist.sample().cpu().numpy()[0]
        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward
        steps += 1

    env.close()
    return dict(
        reward=ep_reward,
        success=info.get("landing_success", False),
        legs=info.get("legs_touching", 0),
        vel=info.get("landing_velocity", np.nan),
        dist=info.get("distance_from_pad", np.nan),
        steps=steps,
    )


def evaluate_model(
    model,
    env,
    n_episodes=100,
    device=torch.device("cpu"),
    num_workers=None,
    T=1024,
    disable_tqdm=True,
):
    """
    Parallelized evaluation of PPO/GRPO model.
    Supports:
      • env: Gym ID string
      • env: (class, kwargs) spec
      • env: already-instantiated environment (will be converted to spec)
    """
    # Convert env to a picklable spec if needed
    if not isinstance(env, (str, tuple)):
        env_spec, _ = env_to_spec(env)
    else:
        env_spec = env

    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    print(f"\n🎯 Evaluating {model.__class__.__name__} for {n_episodes} episodes using {num_workers} processes...")

    args = [(model, env_spec, device, T) for _ in range(n_episodes)]

    ctx = get_context("fork")  # safer on Linux/Mac/Jupyter

    with ctx.Pool(num_workers) as pool:
        iterator = pool.imap_unordered(_evaluate_episode, args)
        results = list(
            tqdm(iterator, total=n_episodes, disable=disable_tqdm)
            if not disable_tqdm else iterator
        )

    rewards = np.array([r["reward"] for r in results])
    successes = np.array([r["success"] for r in results])
    legs = np.array([r["legs"] for r in results])
    velocities = np.array([r["vel"] for r in results])
    distances = np.array([r["dist"] for r in results])
    lengths = np.array([r["steps"] for r in results])

    stats = dict(
        mean_reward=np.mean(rewards),
        std_reward=np.std(rewards),
        success_rate=np.mean(successes),
        mean_legs=np.mean(legs),
        mean_velocity=np.nanmean(velocities),
        mean_distance=np.nanmean(distances),
        mean_length=np.mean(lengths),
    )

    print(
        f"   ✓ Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}\n"
        f"   ✓ Success Rate: {stats['success_rate']:.2%}\n"
        f"   ✓ Legs Touching: {stats['mean_legs']:.2f}\n"
        f"   ✓ Mean Velocity: {stats['mean_velocity']:.3f}\n"
        f"   ✓ Distance from Pad: {stats['mean_distance']:.3f}"
    )
    return stats

