import sys
import math
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import colorize, seeding, EzPickle

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef,
                      polygonShape, revoluteJointDef, contactListener)

WINDOW_W = 1024
WINDOW_H = 768
FPS = 60

SHIP_POLY = [(-5, 0), (0, 5), (5, 0), (5, -20), (-5, -20)]


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        pass

    def EndContact(self, contact):
        pass


class ShipEnv(gym.Env, EzPickle):

    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': FPS}

    def __init__(self):
        EzPickle.__init__(self)
        self.viewer = None
        self.world = Box2D.b2World((0, 0), doSleep=True)
        self.ship = None
        self.reset()

    def step(self, action):
        reward = 0
        done = False
        state = [0, 0]

        self.ship.ApplyForce(
            (0, 10), (self.ship.position[0] + 1, self.ship.position[1]), True)

        self.world.Step(1.0/FPS, 6, 2)
        return np.array(state, dtype=np.float32), reward, done, {}

    def _destroy(self):
        if not self.ship:
            return
        self.world.contactListener = None
        self.world.DestroyBody(self.ship)
        self.ship = None

    def reset(self):
        self._destroy()
        self.world.contactListener = ContactDetector(self)

        self.ship = self.world.CreateDynamicBody(
            position=(100, 100),
            angle=0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=SHIP_POLY),
                density=1.0,
                friction=0.5,
                restitution=0.1
            )
        )
        self.ship.color1 = (1.0, 0, 0)
        self.ship.color2 = (0, 0, 0)

        self.drawlist = [self.ship]
        return self.step(np.array([0, 0], dtype=np.float32))

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.viewer.set_bounds(0, WINDOW_W, 0, WINDOW_H)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                self.viewer.draw_polygon(path, color=obj.color1)
                path.append(path[0])
                self.viewer.draw_polyline(path, color=obj.color2, linewidth=1)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    env = ShipEnv()
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        s, r, done, info = env.step(np.array([0, 0], dtype=np.float32))
        total_reward += r
        if env.render() == False:
            break
        if steps % 20 == 0 or done:
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done:
            break
    env.close()
