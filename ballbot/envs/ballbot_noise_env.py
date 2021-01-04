import logging
import numpy as np

from ballbot.envs.ballbot_env import Ballbot_Env

logger = logging.getLogger(__name__)

class Ballbot_Noise_Env(Ballbot_Env):

    def _compute_observation(self):
        observation = super(Ballbot_Noise_Env,self)._compute_observation()
        return [observation[0] + np.random.normal(0, 0.02),
                observation[1] + np.random.normal(0, 0.02),
                observation[2] + np.random.normal(0, 0.02),
                observation[3] + np.random.normal(0, 0.01),
                observation[4] + np.random.normal(0, 0.01),
                observation[5] + np.random.normal(0, 0.01),
                observation[6] + np.random.normal(0, 0.02),
                observation[7] + np.random.normal(0, 0.02),
                observation[8] + np.random.normal(0, 0.02)
                ]

    def _reset(self):
        observation = super(Ballbot_Noise_Env,self)._reset()
        return [observation[0] + np.random.normal(0, 0.02),
                observation[1] + np.random.normal(0, 0.02),
                observation[2] + np.random.normal(0, 0.02),
                observation[3] + np.random.normal(0, 0.01),
                observation[4] + np.random.normal(0, 0.01),
                observation[5] + np.random.normal(0, 0.01),
                observation[6] + np.random.normal(0, 0.02),
                observation[7] + np.random.normal(0, 0.02),
                observation[8] + np.random.normal(0, 0.02)]
