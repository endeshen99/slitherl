from gym.envs.registration import register

register(
    id='slitherl-v0',
    entry_point='gym_slitherl.envs:SlitherlEnv',
)
register(
    id='slitherl-extrahard-v0',
    entry_point='gym_slitherl.envs:SlitherlExtraHardEnv',
)