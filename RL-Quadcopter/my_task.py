import numpy as np
from physics_sim import PhysicsSim
from math import sqrt
from math import exp

class MyTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_cartesian_distance(self, current, target): 
        """ Compute Cartesian distance between two points """
        return sqrt(sum( [(a - b)**2 for a, b in zip(current, target) ] ) )

    def sigmoid(x):
          return 1 / (1 + exp(-x))
        
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        """ TODO - update the reward function here """
#         current_height = self.sim.pose[2]
#         hf = self.target_pos[2]//2
#         if current_height < hf :
#             hf = hf*(-1.0)
#         else:
#             # hf = hf
#             pass
#         reward = np.tanh(1 - 0.003*self.get_cartesian_distance(self.sim.pose[:3], self.target_pos)) + hf
        
        """ Original reward function in task.py """
#          reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        
#         reward = np.tanh(1 - 0.0003*self.get_cartesian_distance(self.sim.pose[:3], self.target_pos)) + \
#                     ( min(10, self.sim.pose[2:3][0]) - self.target_pos[2])

#       reward = 100*min(10, self.sim.pose[2:3][0]) - self.get_cartesian_distance(self.sim.pose[:3], self.target_pos[:3])
        reward = 1.-.003*(abs(self.sim.pose[:3] - self.target_pos[:3])).sum() + 10*min(self.target_pos[2],self.sim.pose[2])
    
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state