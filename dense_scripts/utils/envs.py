# <dense_script/utils/envs.py>
import math

from typing import Optional, Dict, Any, Tuple, List, Type, Union
import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander
import numpy as np


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
                 velocity_penalty_scale=10.0, leg_bonus_scale=20.0,
                 distance_penalty_scale=10.0, tilt_penalty_scale=10.0, 
                 out_of_bounds_penalty=-5.0, max_episode_steps=1000, **kwargs):
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
        
        # Track episode statistics
        self.episode_steps = 0
        self.total_original_reward = 0.0
        
    def reset(self, **kwargs):
        """Reset the environment and tracking variables."""
        self.episode_steps = 0
        self.total_original_reward = 0.0
        return super().reset(**kwargs)
    
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
            # Check if this was a successful landing or a crash
            # Success condition: both legs touching ground, low speed, and close to landing pad
            # Get position and velocity
            x_position = observation[0]
            vx = observation[2]
            vy = observation[3]
            speed = np.sqrt(vx**2 + vy**2)
            legs_touching = int(self.legs[0].ground_contact) + int(self.legs[1].ground_contact)
            # New success criteria: both legs on ground, speed < 0.01, abs(x) <= 0.2
            is_success = ((legs_touching == 2) and (speed < 0.01) and (abs(x_position) <= 0.2)) or (not self.lander.awake)
            
            if is_success:
                # Successful landing
                sparse_reward += self.success_reward
                info['landing_success'] = True
                info['crash_velocity'] = float(speed)
                info['legs_touching'] = 2
                info['distance_from_pad'] = float(abs(x_position))
                info['out_of_bounds'] = False
            else:
                # Failed landing - calculate penalty based on crash severity
                info['landing_success'] = False
                
                # Get position (observation[0] is x position, landing pad is at x=0)
                x_position = observation[0]
                y_position = observation[1]
                distance_from_pad = np.sqrt(x_position**2 + y_position**2)
                
                # Check if lander flew off screen
                # LunarLander terminates when abs(state[0]) >= 1.0 (see lunar_lander.py line 658)
                # The observation space bounds are [-2.5, 2.5] but termination happens at [-1.0, 1.0]
                # Y position doesn't trigger termination, only x position does
                out_of_bounds = abs(x_position) >= 1.0
                info['out_of_bounds'] = out_of_bounds
                
                # Get velocity at crash (from observation)
                # observation[2] = horizontal velocity, observation[3] = vertical velocity
                vx = observation[2]
                vy = observation[3]
                crash_velocity = np.sqrt(vx**2 + vy**2)
                
                # Get angle/tilt at crash (observation[4] is angle in radians)
                # Angle of 0 means upright, larger angles mean more tilted
                angle = abs(observation[4])
                
                # Check how many legs were touching
                legs_touching = int(self.legs[0].ground_contact) + int(self.legs[1].ground_contact)
                
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
                                    max(0, 1 - x_position**2))
                
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


# dense_scripts/utils/envs.py
import numpy as np
import math
from gymnasium.envs.box2d.lunar_lander import LunarLander
from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef

import numpy as np
from gymnasium.envs.box2d.lunar_lander import (
    LunarLander,
    VIEWPORT_W,
    VIEWPORT_H,
    SCALE,
    LEG_DOWN,
    LEG_AWAY,
    LEG_SPRING_TORQUE,
    LANDER_POLY,
    LEG_W,
    LEG_H,
    INITIAL_RANDOM,
)


class DenseLunarLander(LunarLander):
    """
    Dense-reward LunarLander variant that:
      ✅ Randomizes spawn position and angle (robustly, without breaking Box2D world)
      ✅ Adds diagnostic info at episode end
      ✅ Maintains compatibility with Gymnasium's rendering and physics
    """

    def __init__(
        self,
        randomize_angle=False,
        randomize_angle_range=0.25,
        randomize_pos=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Rebind constants so they persist through pickling
        self.VIEWPORT_W = VIEWPORT_W
        self.VIEWPORT_H = VIEWPORT_H
        self.SCALE = SCALE
        self.LEG_DOWN = LEG_DOWN
        self.LEG_AWAY = LEG_AWAY
        self.LEG_SPRING_TORQUE = LEG_SPRING_TORQUE
        self.LANDER_POLY = LANDER_POLY
        self.LEG_W = LEG_W
        self.LEG_H = LEG_H
        self.INITIAL_RANDOM = INITIAL_RANDOM

        self.randomize_angle = randomize_angle
        self.randomize_angle_range = randomize_angle_range
        self.randomize_pos = randomize_pos

    def reset(self, *, seed=None, options=None):
        # Call parent reset to initialize terrain, world, and contact listener
        obs, info = super().reset(seed=seed, options=options)

        # Safely destroy the existing lander + legs
        if getattr(self, "lander", None):
            self.world.DestroyBody(self.lander)
        if hasattr(self, "legs"):
            for leg in self.legs:
                if leg:
                    self.world.DestroyBody(leg)

        # Define world scale and viewport
        W = self.VIEWPORT_W / self.SCALE
        H = self.VIEWPORT_H / self.SCALE

        # Randomized spawn position and angle
        initial_x = (
            self.np_random.uniform(0.1 * W, 0.9 * W) if self.randomize_pos else W / 2
        )
        # print("initial x", initial_x)
        # print('done')
        # quit()
        initial_y = self.np_random.uniform(H * 0.7, H) if self.randomize_pos else H
        initial_angle = (
            self.np_random.uniform(
                -self.randomize_angle_range, self.randomize_angle_range
            )
            if self.randomize_angle
            else 0.0
        )

        # Create new lander
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=initial_angle,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / self.SCALE, y / self.SCALE) for x, y in self.LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0,
            ),
        )

        self.lander.color1 = (128, 102, 230)
        self.lander.color2 = (77, 77, 128)

        # Apply random initial push to simulate turbulence
        self.lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-self.INITIAL_RANDOM, self.INITIAL_RANDOM),
                self.np_random.uniform(-self.INITIAL_RANDOM, self.INITIAL_RANDOM),
            ),
            True,
        )

        # --- Recreate legs, aligned to body orientation ---
        # ✅ FIX 2: Create legs relative to the lander's rotated position and angle
        c = math.cos(initial_angle)
        s = math.sin(initial_angle)

        self.legs = []
        for i in [-1, +1]:
            # Calculate leg's center position relative to lander's center
            # The leg's *center* is offset horizontally, not the joint
            leg_rel_x = -i * self.LEG_AWAY / self.SCALE
            leg_rel_y = 0.0
    
            # Rotate this relative position by the lander's angle
            leg_world_x = initial_x + (leg_rel_x * c - leg_rel_y * s)
            leg_world_y = initial_y + (leg_rel_x * s + leg_rel_y * c)
    
            # Leg's angle must also be relative to the lander
            leg_angle = initial_angle + (i * 0.05)
    
            leg = self.world.CreateDynamicBody(
                position=(leg_world_x, leg_world_y),
                angle=leg_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(box=(self.LEG_W / self.SCALE, self.LEG_H / self.SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (128, 102, 230)
            leg.color2 = (77, 77, 128)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                # Anchor on lander: (0, 0) in lander's local frame (its center)
                localAnchorA=(0, 0),
                # Anchor on leg: offset from leg's center
                localAnchorB=(i * self.LEG_AWAY / self.SCALE, self.LEG_DOWN / self.SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,
            )
            if i == -1:
                rjd.lowerAngle = +0.4
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.4
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        # Update draw list
        self.drawlist = [self.lander] + self.legs
    
        # ✅ FIX 1: Temporarily disable wind to prevent initial torque from super().step(0)
        was_wind_enabled = self.enable_wind
        self.enable_wind = False
    
        # Advance one step to compute valid observation
        obs, _, _, _, _ = super().step(0 if not self.continuous else np.array([0.0, 0.0]))
    
        # Restore original wind setting
        self.enable_wind = was_wind_enabled
    
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Attach diagnostics
        if terminated or truncated:
            x, y, vx, vy, angle = obs[0], obs[1], obs[2], obs[3], obs[4]
            landing_velocity = float(np.sqrt(vx**2 + vy**2))
            legs_touching = int(self.legs[0].ground_contact) + int(self.legs[1].ground_contact)
            distance_from_pad = float(abs(x))
            landing_success = (legs_touching == 2) and (landing_velocity < 0.1) and (abs(x) < 0.2)

            info.update(
                landing_velocity=landing_velocity,
                legs_touching=legs_touching,
                distance_from_pad=distance_from_pad,
                landing_success=landing_success,
            )

        return obs, reward, terminated, truncated, info



class StatefulLunarLander(DenseLunarLander):
    """
    A wrapper around DenseLunarLander that adds `get_state()` and `set_state()` methods.
    
    This is CRITICAL for Monte Carlo search algorithms, as it allows the
    environment to be "rewound" to a previous state to explore different actions.
    """
    
    def get_state(self) -> Tuple[Any, Any, Any]:
        """
        Saves the full physical and logical state of the environment
        into simple, pickle-able Python types.
        """
        # ✅ FIX: Convert Box2D b2Vec2 objects to plain tuples
        lander_state = (
            (self.lander.position[0], self.lander.position[1]), # <-- Convert to tuple
            self.lander.angle,
            (self.lander.linearVelocity[0], self.lander.linearVelocity[1]), # <-- Convert to tuple
            self.lander.angularVelocity
        )
        legs_state = [
            (
                (leg.position[0], leg.position[1]), # <-- Convert to tuple
                leg.angle,
                (leg.linearVelocity[0], leg.linearVelocity[1]), # <-- Convert to tuple
                leg.angularVelocity
            )
            for leg in self.legs
        ]
        
        game_state = (
            self.game_over,
            self.prev_shaping,
            getattr(self, 'wind_idx', 0),
            getattr(self, 'torque_idx', 0)
        )
        return (lander_state, legs_state, game_state)

    def set_state(self, state: Tuple[Any, Any, Any]):
        """
        Restores the full physical and logical state of the environment.
        """
        lander_state, legs_state, game_state = state
        
        # ✅ FIX: Box2D properties are read-only, so we must set them
        #      using their .Set() method or by assigning a new tuple/list.
        
        pos_tuple, angle, lin_vel_tuple, ang_vel = lander_state
        self.lander.position = pos_tuple
        self.lander.angle = angle
        self.lander.linearVelocity = lin_vel_tuple
        self.lander.angularVelocity = ang_vel
        self.lander.awake = True

        for i, leg_state in enumerate(legs_state):
            pos_tuple, angle, lin_vel_tuple, ang_vel = leg_state
            self.legs[i].position = pos_tuple
            self.legs[i].angle = angle
            self.legs[i].linearVelocity = lin_vel_tuple
            self.legs[i].angularVelocity = ang_vel
            self.legs[i].awake = True

        (self.game_over, self.prev_shaping, 
         wind_idx, torque_idx) = game_state
        
        self.wind_idx = wind_idx
        self.torque_idx = torque_idx