import numpy as np
from abc import ABC, abstractmethod
from gym.spaces import Dict

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        if isinstance(env.observation_space, Dict):
            if 'depth' in env.observation_space.spaces:
                self.batch_ob_shape = (
                    (nenv*nsteps,) + env.observation_space.spaces['depth'].shape,
                    (nenv*nsteps,) + env.observation_space.spaces['pointgoal'].shape
                )
                self.obs = (
                    np.zeros((nenv,) + env.observation_space.spaces['depth'].shape,
                             dtype=env.observation_space.spaces['depth'].dtype.name),
                    np.zeros((nenv,) + env.observation_space.spaces['pointgoal'].shape,
                             dtype=env.observation_space.spaces['pointgoal'].dtype.name)
                )
                obs_reset = env.reset()
                batch_depth = []
                batch_goal = []
                for xb in range(len(obs_reset)):
                    batch_depth.append(obs_reset[xb]['depth'])
                    batch_goal.append(obs_reset[xb]['pointgoal'])
                batch_depth = np.array(batch_depth)
                batch_goal = np.array(batch_goal)
                self.obs[0][:] = batch_depth
                self.obs[1][:] = batch_goal
            else:
                self.batch_ob_shape = (
                    (nenv*nsteps,) + env.observation_space.spaces['rgb'].shape,
                    (nenv*nsteps,) + env.observation_space.spaces['pointgoal'].shape
                )
                self.obs = (
                    np.zeros((nenv,) + env.observation_space.spaces['rgb'].shape,
                             dtype=env.observation_space.spaces['rgb'].dtype.name),
                    np.zeros((nenv,) + env.observation_space.spaces['pointgoal'].shape,
                             dtype=env.observation_space.spaces['pointgoal'].dtype.name)
                )
                obs_reset = env.reset()
                batch_rgb = []
                batch_goal = []
                for xb in range(len(obs_reset)):
                    batch_rgb.append(obs_reset[xb]['rgb'])
                    batch_goal.append(obs_reset[xb]['pointgoal'])
                batch_rgb = np.array(batch_rgb)
                batch_goal = np.array(batch_goal)
                self.obs[0][:] = batch_rgb
                self.obs[1][:] = batch_goal
        else:
            self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
            self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
            self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError

