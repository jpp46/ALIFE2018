import math
import numpy as np
import pyrosim

from environment import Environment

HEIGHT = 0.3
EPS = 0.05
WHEEL_RADIUS = 0.1
SPEED = 10


class Robot(object):
    def __init__(self, sim, wts):
        main_body = sim.send_box(x=0, y=0, z=HEIGHT+EPS,
                                 length=HEIGHT, width=HEIGHT,
                                 height=EPS*2.0, mass=1)

        self.light = sim.send_light_sensor(main_body)
        self.position = sim.send_position_sensor(main_body)
        self.wts=wts

        # id arrays
        thighs = [0] * 4
        shins = [0] * 4
        hips = [0] * 4
        knees = [0] * 4
        swivels = [0] * 4
        saxles = [0] * 4
        wheels = [0] * 4
        waxles = [0] * 4
        foot_sensors = [0] * 4
        sensor_neurons = [0] * 5
        motor_neurons = [0] * 8

        sensor_neurons[-1] = sim.send_sensor_neuron(self.light)

        delta = float(math.pi) / 2.0

        for i in range(4):
            theta = delta*i
            x_pos = math.cos(theta)*HEIGHT
            y_pos = math.sin(theta)*HEIGHT

            thighs[i] = sim.send_cylinder(x=x_pos, y=y_pos, z=HEIGHT+EPS,
                                          r1=x_pos, r2=y_pos, r3=0,
                                          length=HEIGHT, radius=EPS, capped=True
                                          )

            hips[i] = sim.send_hinge_joint(main_body, thighs[i],
                                           x=x_pos/2.0, y=y_pos/2.0, z=HEIGHT+EPS,
                                           n1=-y_pos, n2=x_pos, n3=0,
                                           lo=0, hi=0,
                                           speed=1.0)

            x_pos2 = math.cos(theta)*1.5*HEIGHT
            y_pos2 = math.sin(theta)*1.5*HEIGHT

            shins[i] = sim.send_cylinder(x=x_pos2, y=y_pos2, z=(HEIGHT+EPS)/2.0,
                                         r1=0, r2=0, r3=1,
                                         length=HEIGHT, radius=EPS,
                                         mass=1., capped=True)

            knees[i] = sim.send_hinge_joint(thighs[i], shins[i],
                                            x=x_pos2, y=y_pos2, z=HEIGHT+EPS,
                                            n1=-y_pos, n2=x_pos, n3=0,
                                            lo=0, hi=0)

            swivels[i] = sim.send_sphere(
                x=x_pos2, y=y_pos2, z=(HEIGHT/2 - EPS) - WHEEL_RADIUS, radius=WHEEL_RADIUS)

            saxles[i] = sim.send_hinge_joint(first_body_id=swivels[i],
                                                second_body_id=shins[i], x=x_pos2,
                                                y=y_pos2, z=(HEIGHT/2 - EPS) - WHEEL_RADIUS,
                                                n1=0, n2=1, n3=0,
                                                position_control=False,
                                                speed=SPEED)

            motor_neurons[i] = sim.send_motor_neuron(joint_id=saxles[i])

            wheels[i] = sim.send_sphere(
                x=x_pos2, y=y_pos2, z=(HEIGHT/2 - EPS) - WHEEL_RADIUS, radius=WHEEL_RADIUS)

            waxles[i] = sim.send_hinge_joint(first_body_id=wheels[i],
                                                second_body_id=swivels[i], x=x_pos2,
                                                y=y_pos2, z=(HEIGHT/2 - EPS) - WHEEL_RADIUS,
                                                n1=1, n2=0, n3=0,
                                                position_control=False,
                                                speed=SPEED)

            motor_neurons[i+4] = sim.send_motor_neuron(waxles[i])
            foot_sensors[i] = sim.send_touch_sensor(wheels[i])
            sensor_neurons[i] = sim.send_sensor_neuron(foot_sensors[i])

        for i in range(5):
            for j in range(8):
                sim.send_synapse(sensor_neurons[i], motor_neurons[j], weight=wts[i, j])

    def evaluate(self, sim, env):
        env.send_to(sim)
        sim.start()

    def eval_fitness(self, sim, env):
        sim.wait_to_finish()
        x = sim.get_sensor_data(self.position, svi=0)[-1]
        y = sim.get_sensor_data(self.position, svi=1)[-1]
        distance = np.sqrt((env.x - x)**2 + (env.y - y)**2)

        sensor_data = sim.get_sensor_data(self.light)
        fitness = sensor_data[-1]

        del sim
        return fitness, distance


'''def init_evnironments(num):
    envs = []
    for i in range(num):
        envs.append(Environment(i))
    return np.array(envs)

def new_sim(pp=True, pb=True, t=1000):
    return pyrosim.Simulator(play_paused=pp, play_blind=pb, eval_time=t)

def compute_fitness(population, env):
    sims = []; fits = []; dist = [];
    for i in range((2 // 2)):
        shift = i*10
        for j in range(2):
            sims.append(new_sim(pb=False, pp=True))
            robot = Robot(sims[j+shift], population[j+shift])
            robot.evaluate(sims[j+shift], env)
            fitness, distance = robot.eval_fitness(sims[j+shift], env)
            fits.append(fitness); dist.append(distance)

    return np.array(fits), np.array(dist)

if __name__ == "__main__":

    envs = init_evnironments(4)
    parents = np.random.random((2, 5, 8)) * 2.0 - 1.0
    parent_fits, parent_dist = compute_fitness(parents, envs[0])
    parent_fits, parent_dist = compute_fitness(parents, envs[2])'''
