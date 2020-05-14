import gym
from gym import error, spaces, utils
from gym.utils import colorize, seeding, EzPickle
import pyglet
from pyglet import gl

WINDOW_W = 1024
WINDOW_H = 768
FPS = 60

class ShipEnv(gym.Env, EzPickle):
    
    metadata = {'render.modes': ['human'],
            'video.frames_per_second' : FPS}

    def __init__(self):
        EzPickle.__init__(self)
        self.viewer = None
    
    def step(self, action):
        pass
    
    def _destroy(self):
        pass

    def reset(self):
        self._destroy()

    def render(self, mode='human'):
        assert mode in ['human']
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.transform = rendering.Transform()
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()
        win.clear()
        pixel_scale = 1
        if hasattr(win.context, '_nscontext'):
            pixel_scale = win.context._nscontext.view().backingScaleFactor()  
        VP_W = int(pixel_scale * WINDOW_W)
        VP_H = int(pixel_scale * WINDOW_H) 

        t = self.transform
        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()

        gl.glBegin(gl.GL_QUADS)
        gl.glVertex2f(100, 100)
        gl.glVertex2f(200, 100)
        gl.glVertex2f(200, 200)
        gl.glVertex2f(100, 200)
        gl.glEnd()

        t.disable()
        
        if mode == 'human':
            win.flip()
            return self.viewer.isopen
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]
        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__=="__main__":
    
    env = ShipEnv()
    env.render()
    isopen = True
    while isopen:
        env.reset()
        while True:
            isopen = env.render()
            if isopen == False:
                break
    env.close()

