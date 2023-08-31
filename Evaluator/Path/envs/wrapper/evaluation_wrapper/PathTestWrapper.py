import gym, sys, math, numpy

class PathTestWrapper(gym.Wrapper):
    def __init__(self, env ,cfg):
        super(PathTestWrapper, self).__init__(env)
        self.cur_num = 0
        self.test_num = cfg["init_pose_bag_episodes"]
        self.info = {}
        self.reset_kwargs = None
        self.obs_list = []
        self.robot_pose = None
        self.goal_pose = None
        

    def reset(self, **kwargs):
        self.reset_kwargs = kwargs
        state, reward, done, info = self.env.reset(**kwargs)
        self.obs_list = self._from_obs_msgs(info['obs_msgs'])
        self.robot_pose, self.goal_pose = self._from_robot_msgs(info['robot_msgs'])
        info['obs_list'] = self.obs_list
        info['robot_pose'] = self.robot_pose
        info['goal_pose'] = self.goal_pose
        return state, reward, done, info
    
    def step(self, paths:dict):
        if self.cur_num < self.test_num:
            for key in paths.keys():
                info = self._statistic(paths[key])
                if self.info.get(key, None) is None:
                    self.info[key] = info
                else:
                    self.info[key] += info
            self.cur_num += 1
            return self.reset(**self.reset_kwargs)
        else:
            self._print_info()
            sys.exit()

    def _from_obs_msgs(self, obs_msgs):
        obs_list = []
        for obs in obs_msgs:
            position = [obs.init_pose.position.x, obs.init_pose.position.y]
            radius = obs.size[2] if obs.shape == 'circle' else math.sqrt(obs.size[0]**2+obs.size[2]**2)
            obs_list.append(position + [radius,])
        return obs_list

    def _from_robot_msgs(self, robot_msgs):
        robot = robot_msgs[0]
        position = [robot.init_pose.position.x, robot.init_pose.position.y]
        goal_position = [robot.goal.x, robot.goal.y]
        yaw = math.atan2(robot.goal.y - robot.init_pose.position.y, robot.goal.x - robot.init_pose.position.x) % (2*math.pi)
        return position + [yaw,], goal_position + [yaw,]

    def _statistic(self, path:list):
        length = []
        obs_dist = []
        curve = []
        for i in range(1, len(path)):
            length.append(self._dist(path[i], path[i-1]))
            obs_dist.append(self._obs_dist(path[i]))
            if i < len(path)-1:
                curve.append(self._curve(path[i-1], path[i], path[i+1]))
        sum_length = sum(length)
        goal_forward = self._dist(self.goal_pose, path[0]) - self._dist(self.goal_pose, path[-1])
        forward_per_length = goal_forward / sum_length if sum_length != 0 else 0
        min_obs_dist = min(obs_dist)
        mean_curve = sum(curve) / len(curve)

        return numpy.array([forward_per_length, min_obs_dist, mean_curve])

    def _print_info(self):
        for key in self.info.keys():
            print("{}:\nForward_per_Len:{}\nMin_Obs_Dist:{}\nCurve:{}\n".format(key, self.info[key][0]/self.test_num, self.info[key][1]/self.test_num, self.info[key][2]/self.test_num))

    def _dist(self, point1, point2):
        return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    
    def _obs_dist(self, point):
        dist = [self._dist(point, obs) for obs in self.obs_list]
        dist = [dist[i]-self.obs_list[i][2] for i in range(len(dist))]
        return min(dist)
    
    def _curve(self, point1, point2, point3):
        dy = math.tan(point2[2])
        ddy = math.fabs((math.tan(point3[2])-math.tan(point1[2]))/(point3[0]-point1[0])) if point3[0] != point1[0] else 0
        return ddy/(1+dy**2)**1.5