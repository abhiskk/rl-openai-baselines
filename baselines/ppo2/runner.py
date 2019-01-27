import numpy as np
from baselines.common.runners import AbstractEnvRunner
from gym.spaces import Dict

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self):
        # Here, we init the lists that will contain the mb of experiences

        if isinstance(self.env.observation_space, Dict):
            if 'depth' in self.env.observation_space.spaces:
                mb_depth = []
            else:
                mb_rgb = []
            mb_goal = []
            multimodal = True
        else:
            mb_obs = []
            multimodal = False

        mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            if multimodal:
                if 'depth' in self.env.observation_space.spaces:
                    mb_depth.append(self.obs[0].copy())
                else:
                    mb_rgb.append(self.obs[0].copy())
                mb_goal.append(self.obs[1].copy())
            else:
                mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            if multimodal:
                multimodal_obs, rewards, self.dones, infos = self.env.step(actions)

                if 'depth' in self.env.observation_space.spaces:
                    temp_batch_depth = []
                else:
                    temp_batch_rgb = []
                temp_batch_goal = []

                for ix in range(len(multimodal_obs)):
                    if 'depth' in self.env.observation_space.spaces:
                        temp_batch_depth.append(multimodal_obs[ix]['depth'])
                    else:
                        temp_batch_rgb.append(multimodal_obs[ix]['rgb'])
                    temp_batch_goal.append(multimodal_obs[ix]['pointgoal'])

                if 'depth' in self.env.observation_space.spaces:
                    temp_batch_depth = np.array(temp_batch_depth)
                else:
                    temp_batch_rgb = np.array(temp_batch_rgb)
                temp_batch_goal = np.array(temp_batch_goal)

                if 'depth' in self.env.observation_space.spaces:
                    self.obs[0][:] = temp_batch_depth
                else:
                    self.obs[0][:] = temp_batch_rgb
                self.obs[1][:] = temp_batch_goal
            else:
                self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        if multimodal:
            if 'depth' in self.env.observation_space.spaces:
                mb_depth = np.array(mb_depth, dtype=self.obs[0].dtype)
            else:
                mb_rgb = np.array(mb_rgb, dtype=self.obs[0].dtype)
            mb_goal = np.array(mb_goal, dtype=self.obs[1].dtype)
        else:
            mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        if multimodal:
            if 'depth' in self.env.observation_space.spaces:
                return (*map(sf01, (mb_depth, mb_goal, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                        mb_states, epinfos)
            else:
                return (*map(sf01, (mb_rgb, mb_goal, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                        mb_states, epinfos)
        else:
            return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
                mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


