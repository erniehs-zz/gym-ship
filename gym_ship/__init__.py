from gym.envs.registration import register

register(id='ship-v0',
        entry_point='gym_ship.envs:ShipEnv',
)
