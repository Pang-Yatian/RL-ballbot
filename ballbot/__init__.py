import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id = 'ballbot-v0',
    entry_point = 'ballbot.envs:Ballbot_Env'
)
register(
    id = 'ballbot_noise-v0',
    entry_point = 'ballbot.envs:Ballbot_Noise_Env'
)