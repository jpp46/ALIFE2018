HEIGHT = 0.3
EPS = 0.05


class Environment(object):
    def __init__(self, idx):
        self.idx = idx
        self.l = HEIGHT
        self.w = HEIGHT
        self.h = HEIGHT
        if self.idx == 0:
            self.x = 30 * HEIGHT
            self.y = 0
            self.z = HEIGHT / 2.0
        if self.idx == 1:
            self.x = 0
            self.y = 30 * HEIGHT
            self.z = HEIGHT / 2.0
        if self.idx == 2:
            self.x = -30 * HEIGHT
            self.y = 0
            self.z = HEIGHT / 2.0
        if self.idx == 3:
            self.x = 0
            self.y = -30 * HEIGHT
            self.z = HEIGHT / 2.0

    def send_to(self, sim):
        light_source = sim.send_box(x=self.x, y=self.y, z=self.z,
                           length=self.l, width=self.w, height=self.h,
                           r=0.5, g=0.5, b=0.5)
        sim.send_light_source(light_source)
