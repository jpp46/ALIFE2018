import math
import numpy as np
import pyrosim

HEIGHT = 0.3
EPS = 0.05


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
                                           lo=-math.pi/4.0, hi=math.pi/4.0,
                                           speed=1.0)

            motor_neurons[i] = sim.send_motor_neuron(joint_id=hips[i])

            x_pos2 = math.cos(theta)*1.5*HEIGHT
            y_pos2 = math.sin(theta)*1.5*HEIGHT

            shins[i] = sim.send_cylinder(x=x_pos2, y=y_pos2, z=(HEIGHT+EPS)/2.0,
                                         r1=0, r2=0, r3=1,
                                         length=HEIGHT, radius=EPS,
                                         mass=1., capped=True)

            knees[i] = sim.send_hinge_joint(thighs[i], shins[i],
                                            x=x_pos2, y=y_pos2, z=HEIGHT+EPS,
                                            n1=-y_pos, n2=x_pos, n3=0,
                                            lo=-math.pi/4.0, hi=math.pi/4.0)

            motor_neurons[i+4] = sim.send_motor_neuron(knees[i])
            foot_sensors[i] = sim.send_touch_sensor(shins[i])
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
