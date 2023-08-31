import math, numpy

class PurePursuit:
    def __init__(self, cfg):
        self.dt = cfg['control_hz']

    @classmethod
    def vector_to_location(cls, vector):
        a = vector[0][0]
        b = vector[0][1]
        yaw = vector[0][2]
        x = -a*math.cos(yaw)-b*math.sin(yaw)
        y = a*math.sin(yaw)-b*math.cos(yaw)
        return [x, y, (-yaw)%(2*math.pi)]
    
    @classmethod
    def distance(cls, location1, location2 = [0., 0.]):
        return math.sqrt((location1[0]-location2[0])**2+(location1[1]-location2[1])**2)

    @classmethod
    def angle_from_negative_pi_to_pi(cls, angle):
            if angle > math.pi:
                angle = angle - 2*math.pi*math.ceil(angle/(2*math.pi))
            elif angle < -math.pi:
                angle = angle - 2*math.pi*math.floor(angle/(2*math.pi))
            return angle

    def reset(self):
        pass
    
    def action(self, path, state):
        location = PurePursuit.vector_to_location(state[1])
        r = PurePursuit.distance(path[0], location)
        theta = PurePursuit.angle_from_negative_pi_to_pi(path[0][2] - location[2])
        w = theta / self.dt
        v = r / self.dt * (theta/2) / math.sin(theta/2) if theta != 0.0 else r / self.dt

        return numpy.array([[v, w]]) 