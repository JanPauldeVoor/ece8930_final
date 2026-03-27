from gymnasium.envs.registration import register

register(
    id='gym_so101/SO101Sorting',
    entry_point='gym_so101.env:SO101Env',
    max_episode_steps=300,
    # Even after seeding, the rendered observations are slightly different,
    # so we set `nondeterministic=True` to pass `check_env` tests
    nondeterministic=True,
    kwargs={"obs_type": "so101_pixels_agent_pos", "task": "SO101Sorting"},
)