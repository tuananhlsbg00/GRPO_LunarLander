"""
Sparse Reward Lunar Lander Environment

This module implements a sparse reward version of the LunarLander environment
by inheriting from gymnasium's LunarLander and modifying the reward structure.

Instead of receiving dense rewards at each step, the agent only receives:
- A large positive reward (+100) for successfully landing
- Penalties for crashing based on velocity, distance from pad, and leg contact
- No step penalty (only terminal rewards/penalties)
"""

import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class EnvConfig:
    """Configuration for the SparseLunarLander environment."""
    
    soft_success_condition: bool = False
    random_initial_position: bool = True
    soft_crash_reward: float = 100.0
    success_reward: float = 200.0
    velocity_penalty_scale: float = 40.0
    leg_bonus_scale: float = 20.0
    distance_penalty_scale: float = 30.0
    tilt_penalty_scale: float = 15.0
    out_of_bounds_penalty: float = -10.0
    max_episode_steps: int = 1000
    
    def to_env_kwargs(self) -> Dict[str, Any]:
        """Convert to keyword arguments for SparseLunarLander constructor."""
        return {
            'soft_success_condition': self.soft_success_condition,
            'random_initial_position': self.random_initial_position,
            'soft_crash_reward': self.soft_crash_reward,
            'success_reward': self.success_reward,
            'velocity_penalty_scale': self.velocity_penalty_scale,
            'leg_bonus_scale': self.leg_bonus_scale,
            'distance_penalty_scale': self.distance_penalty_scale,
            'tilt_penalty_scale': self.tilt_penalty_scale,
            'out_of_bounds_penalty': self.out_of_bounds_penalty,
            'max_episode_steps': self.max_episode_steps,
        }


class SparseLunarLander(LunarLander):
    """
    Sparse reward version of LunarLander by directly inheriting from LunarLander.
    
    The original LunarLander provides dense rewards at each step based on:
    - Distance to landing pad
    - Velocity
    - Angle
    - Leg contact
    - Fuel usage
    
    This sparse version only provides rewards at episode termination:
    - +100 for successful landing (both legs on ground, stable)
    - Penalty for crashes, scaled by:
      * Crash velocity (softer crashes get less penalty)
      * Distance from landing pad (farther = worse penalty)
      * Number of legs touching (landing on legs is better than body crash)
    """
    
    def __init__(self, success_reward=200.0, soft_crash_reward=100.0,
                 velocity_penalty_scale=40.0, leg_bonus_scale=20.0,
                 distance_penalty_scale=30.0, tilt_penalty_scale=15.0,
                 out_of_bounds_penalty=-10.0, max_episode_steps=1000,
                 random_initial_position: bool = True, 
                 soft_success_condition: bool = False, **kwargs):
        """
        Initialize the sparse reward LunarLander.
        
        Args:
            success_reward: Reward for successful landing (default: 200.0)
            velocity_penalty_scale: Scale for velocity-based penalty (default: 40.0)
            leg_bonus_scale: Bonus reduction for each leg touching (default: 25.0)
            distance_penalty_scale: Scale for distance-from-pad penalty (default: 40.0)
            tilt_penalty_scale: Scale for tilt/angle penalty (default: 40.0)
            out_of_bounds_penalty: Additional penalty for flying off screen (default: -40.0)
            max_episode_steps: Maximum number of steps per episode (default: 1000)
            random_initial_position: If True, randomize starting position and angle (default: True)
            soft_success_condition: If True, also count slow, stable landings near pad as success (default: False)
            **kwargs: Additional arguments passed to LunarLander constructor
        """
        super().__init__(**kwargs)
        
        self.success_reward = success_reward
        self.soft_crash_reward = soft_crash_reward
        self.velocity_penalty_scale = velocity_penalty_scale
        self.leg_bonus_scale = leg_bonus_scale
        self.distance_penalty_scale = distance_penalty_scale
        self.tilt_penalty_scale = tilt_penalty_scale
        self.out_of_bounds_penalty = out_of_bounds_penalty
        self.max_episode_steps = max_episode_steps
        self.random_initial_position = random_initial_position
        self.soft_success_condition = soft_success_condition
        
        # Track episode statistics
        self.episode_steps = 0
        self.total_original_reward = 0.0
        
    def reset(self, **kwargs):
        """Reset the environment and tracking variables."""
        self.episode_steps = 0
        self.total_original_reward = 0.0
        
        # Call parent reset first
        obs, info = super().reset(**kwargs)

        if self.random_initial_position:
            # Randomize initial position and angle
            # x: -0.6 to 0.6 (normalized coordinates)
            # y: 1.0 to 1.4 (normalized coordinates)
            # angle: -1 to 1 radians
            
            # Convert normalized coordinates to world coordinates
            # x_normalized = (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
            # So: pos.x = x_normalized * (VIEWPORT_W / SCALE / 2) + (VIEWPORT_W / SCALE / 2)
            #           = x_normalized * 10 + 10
            
            # y_normalized = (pos.y - (helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2)
            # So: pos.y = y_normalized * (VIEWPORT_H / SCALE / 2) + (helipad_y + LEG_DOWN / SCALE)
            #           = y_normalized * 6.667 + (3.333 + 0.6)
            
            x_normalized = self.np_random.uniform(-0.6, 0.6)
            y_normalized = self.np_random.uniform(1.0, 1.4)
            random_angle = self.np_random.uniform(-1.0, 1.0)
            
            # Convert to world coordinates
            x_world = x_normalized * 10.0 + 10.0
            y_world = y_normalized * 6.667 + (self.helipad_y + 18.0 / 30.0)
            
            # Set the lander's position and angle
            self.lander.position = (x_world, y_world)
            self.lander.angle = random_angle
            
            # Reset velocities to zero (optional, or you could randomize these too)
            self.lander.linearVelocity = (0, 0)
            self.lander.angularVelocity = 0
            
            # Update legs positions to match the lander
            for i, leg_offset in enumerate([-1, +1]):
                self.legs[i].position = (x_world - leg_offset * 20.0 / 30.0, y_world)
                self.legs[i].angle = random_angle + leg_offset * 0.05
            
            # Recalculate observation with new positions
            pos = self.lander.position
            vel = self.lander.linearVelocity
            
            obs = np.array([
                (pos.x - 10.0) / 10.0,  # x normalized
                (pos.y - (self.helipad_y + 18.0 / 30.0)) / 6.667,  # y normalized
                vel.x * 10.0 / 50.0,  # vx
                vel.y * 10.0 / 50.0,  # vy
                self.lander.angle,  # angle
                20.0 * self.lander.angularVelocity / 50.0,  # angular velocity
                1.0 if self.legs[0].ground_contact else 0.0,
                1.0 if self.legs[1].ground_contact else 0.0,
            ], dtype=np.float32)
        
        return obs, info
    
    def step(self, action):
        """
        Override step method to provide sparse rewards.
        
        Args:
            action: The action to take
            
        Returns:
            observation, sparse_reward, terminated, truncated, info
        """
        # Call parent's step method to get original behavior
        observation, original_reward, terminated, truncated, info = super().step(action)
        
        # Track original reward for comparison
        self.total_original_reward += original_reward
        self.episode_steps += 1
        
        # Check if we've exceeded the step limit
        if self.episode_steps >= self.max_episode_steps:
            truncated = True
            info['TimeLimit.truncated'] = True
        
        # Calculate sparse reward - no reward during episode, only at termination
        sparse_reward = 0.0
        
        if terminated or truncated:
            # Get position and velocity for success/failure detection
            x_position = observation[0]
            y_position = observation[1]
            vx = observation[2]
            vy = observation[3]
            speed = np.sqrt(vx**2 + vy**2)
            
            # Check how many legs were touching
            legs_touching = int(self.legs[0].ground_contact) + int(self.legs[1].ground_contact)
            
            # Determine success condition based on soft_success_condition flag
            if self.soft_success_condition:
                # Soft success: either awake==False OR (both legs down, low speed, near pad)
                landing_success = (not self.lander.awake) or (
                    (legs_touching == 2) and 
                    (speed < 0.05) and 
                    (abs(x_position) <= 0.2)
                )
            else:
                # Standard success: only awake==False
                landing_success = not self.lander.awake
            
            if landing_success:
                # Successful landing
                sparse_reward += self.success_reward
                info['landing_success'] = True
                info['legs_touching'] = legs_touching
                info['out_of_bounds'] = False
                info['soft_success_mode'] = self.soft_success_condition
            else:
                # Failed landing - calculate penalty based on crash severity
                info['landing_success'] = False
                info['soft_success_mode'] = self.soft_success_condition
                
                # Get position (already calculated above)
                distance_from_pad = np.sqrt(x_position**2 + y_position**2)
                
                # Check if lander flew off screen
                # LunarLander terminates when abs(state[0]) >= 1.0 (see lunar_lander.py line 658)
                # The observation space bounds are [-2.5, 2.5] but termination happens at [-1.0, 1.0]
                # Y position doesn't trigger termination, only x position does
                out_of_bounds = abs(x_position) >= 1.0
                info['out_of_bounds'] = out_of_bounds
                
                # Get velocity at crash (already calculated above as vx, vy, speed)
                crash_velocity = speed
                
                # Get angle/tilt at crash (observation[4] is angle in radians)
                # Angle of 0 means upright, larger angles mean more tilted
                angle = abs(observation[4])
                
                # legs_touching already calculated above
                
                # Calculate velocity-based penalty (continuous)
                # Higher velocity = worse penalty
                velocity_penalty = -self.velocity_penalty_scale * crash_velocity
                
                # Calculate distance-based penalty (continuous)
                # Farther from landing pad = worse penalty
                # Landing pad is at x=0, typical crashes are within [-1, 1] range
                distance_penalty = -self.distance_penalty_scale * distance_from_pad
                
                # Calculate tilt-based penalty (continuous)
                # More tilted = worse penalty
                tilt_penalty = -self.tilt_penalty_scale * angle
                
                # Reduce penalty for each leg touching (bonus for landing on legs)
                leg_bonus = self.leg_bonus_scale * legs_touching
                
                # Add extra penalty if flew off screen
                oob_penalty = self.out_of_bounds_penalty if out_of_bounds else 0.0
                
                # Partial success reward for soft crashes with both legs touching near the pad
                # Rewards crashes that are almost successful landings
                soft_crash_reward = (self.soft_crash_reward * int(legs_touching == 2) *
                                    max(0, 1 - np.sqrt(crash_velocity)) * 
                                    max(0, 1 - abs(x_position)))
                
                # Total crash penalty
                crash_penalty = velocity_penalty + distance_penalty + tilt_penalty + leg_bonus + oob_penalty + soft_crash_reward
                
                sparse_reward += crash_penalty
                
                # Store crash details in info
                info['crash_velocity'] = float(crash_velocity)
                info['crash_angle'] = float(angle)
                info['distance_from_pad'] = float(distance_from_pad)
                info['legs_touching'] = legs_touching
                info['velocity_penalty'] = float(velocity_penalty)
                info['distance_penalty'] = float(distance_penalty)
                info['tilt_penalty'] = float(tilt_penalty)
                info['leg_bonus'] = float(leg_bonus)
                info['out_of_bounds_penalty'] = float(oob_penalty)
                info['soft_crash_reward'] = float(soft_crash_reward)
                info['crash_penalty'] = float(crash_penalty)
            
            # Add statistics to info
            info['sparse_reward'] = sparse_reward
            info['original_total_reward'] = self.total_original_reward
            info['episode_length'] = self.episode_steps
        
        return observation, sparse_reward, terminated, truncated, info


# Convenience function to create the environment
def make_sparse_lunar_lander(**kwargs):
    """
    Create a sparse reward Lunar Lander environment.
    
    Args:
        **kwargs: Additional arguments (success_reward, crash_penalty, step_penalty, 
                  render_mode, continuous, gravity, enable_wind, etc.)
        
    Returns:
        SparseLunarLander environment
        
    Example:
        env = make_sparse_lunar_lander(render_mode='human', continuous=False)
        env = make_sparse_lunar_lander(success_reward=200, crash_penalty=-200)
    """
    return SparseLunarLander(**kwargs)


def test_basic():
    """Test basic functionality without rendering."""
    print("=== Testing Basic Functionality ===")
    env = make_sparse_lunar_lander()
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    num_episodes = 3
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            if done:
                success = info.get('landing_success', False)
                original_reward = info.get('original_total_reward', 0)
                
                print(f"Episode finished in {steps} steps")
                print(f"Landing success: {success}")
                
                if not success:
                    crash_vel = info.get('crash_velocity', 0)
                    crash_angle = info.get('crash_angle', 0)
                    distance = info.get('distance_from_pad', 0)
                    legs = info.get('legs_touching', 0)
                    vel_penalty = info.get('velocity_penalty', 0)
                    dist_penalty = info.get('distance_penalty', 0)
                    tilt_penalty = info.get('tilt_penalty', 0)
                    leg_bonus = info.get('leg_bonus', 0)
                    oob = info.get('out_of_bounds', False)
                    oob_penalty = info.get('out_of_bounds_penalty', 0)
                    crash_penalty = info.get('crash_penalty', 0)
                    
                    print(f"  Crash velocity: {crash_vel:.3f}")
                    print(f"  Crash angle (tilt): {crash_angle:.3f} rad ({np.degrees(crash_angle):.1f}°)")
                    print(f"  Distance from pad: {distance:.3f}")
                    print(f"  Legs touching: {legs}")
                    print(f"  Out of bounds: {oob}")
                    print(f"  Velocity penalty: {vel_penalty:.2f}")
                    print(f"  Distance penalty: {dist_penalty:.2f}")
                    print(f"  Tilt penalty: {tilt_penalty:.2f}")
                    print(f"  Leg bonus: {leg_bonus:.2f}")
                    print(f"  Out of bounds penalty: {oob_penalty:.2f}")
                    print(f"  Total crash penalty: {crash_penalty:.2f}")
                
                print(f"Sparse total reward: {total_reward:.2f}")
                print(f"Original total reward: {original_reward:.2f}")
    
    env.close()
    print("\nBasic test completed!")


def test_with_rendering():
    """Test with visual rendering (opens a pygame window)."""
    print("\n=== Testing with Rendering ===")
    print("This will open a pygame window showing the lunar lander.")
    print("Close the window or wait for episodes to complete.\n")
    
    # Create environment with human rendering
    env = make_sparse_lunar_lander(render_mode='human')
    
    num_episodes = 2
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n--- Episode {episode + 1} (with rendering) ---")
        
        while not done:
            # Use random actions
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            # The render() is called automatically in step() when render_mode='human'
            
            if done:
                success = info.get('landing_success', False)
                original_reward = info.get('original_total_reward', 0)
                
                print(f"Episode finished in {steps} steps")
                print(f"Landing success: {success}")
                
                if not success:
                    crash_vel = info.get('crash_velocity', 0)
                    crash_angle = info.get('crash_angle', 0)
                    distance = info.get('distance_from_pad', 0)
                    legs = info.get('legs_touching', 0)
                    oob = info.get('out_of_bounds', False)
                    crash_penalty = info.get('crash_penalty', 0)
                    oob_str = " [OUT OF BOUNDS]" if oob else ""
                    print(f"  Crash: vel={crash_vel:.3f}, angle={np.degrees(crash_angle):.1f}°, dist={distance:.3f}, legs={legs}, penalty={crash_penalty:.2f}{oob_str}")
                
                print(f"Sparse total reward: {total_reward:.2f}")
                print(f"Original total reward: {original_reward:.2f}")
                
                # Brief pause between episodes
                import time
                time.sleep(1)
    
    env.close()
    print("\nRendering test completed!")


def test_with_heuristic():
    """Test with the built-in heuristic controller for better landing."""
    print("\n=== Testing with Heuristic Controller ===")
    print("Using smart controller instead of random actions.\n")
    
    from gymnasium.envs.box2d.lunar_lander import heuristic
    
    env = make_sparse_lunar_lander(render_mode='human')
    
    num_episodes = 2
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n--- Episode {episode + 1} (heuristic control) ---")
        
        while not done:
            # Use heuristic instead of random actions
            action = heuristic(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            if done:
                success = info.get('landing_success', False)
                original_reward = info.get('original_total_reward', 0)
                
                print(f"Episode finished in {steps} steps")
                print(f"Landing success: {success}")
                
                if not success:
                    crash_vel = info.get('crash_velocity', 0)
                    crash_angle = info.get('crash_angle', 0)
                    distance = info.get('distance_from_pad', 0)
                    legs = info.get('legs_touching', 0)
                    oob = info.get('out_of_bounds', False)
                    crash_penalty = info.get('crash_penalty', 0)
                    oob_str = " [OUT OF BOUNDS]" if oob else ""
                    print(f"  Crash: vel={crash_vel:.3f}, angle={np.degrees(crash_angle):.1f}°, dist={distance:.3f}, legs={legs}, penalty={crash_penalty:.2f}{oob_str}")
                
                print(f"Sparse total reward: {total_reward:.2f}")
                print(f"Original total reward: {original_reward:.2f}")
                
                import time
                time.sleep(1)
    
    env.close()
    print("\nHeuristic test completed!")


