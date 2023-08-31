import pymongo, numpy, logging, struct, random, time, os, rosbag, math
from typing import List
from genpy import Time
from tf.transformations import quaternion_matrix, euler_from_matrix

class BagProcess:
    def __init__(self, bag_dir, sum, rank, id_bias=0):
        self.bag_dir = bag_dir
        self.client = pymongo.MongoClient('mongodb://localhost:27017/')
        self.database = self.client['DataBase']
        self.collection = self.database['Collection']
        self.sum = sum
        self.node_id = rank
        self.id_bias = id_bias

        # SCAND robot max speed is 2m/s
        # we need to cut a half of dt and time to make max speed become 1m/s
        self.time = 5.0
        self.dt = 0.125

        self.odom_topic = '/jackal_velocity_controller/odom'
        self.laser_topic = '/velodyne_2dscan_high_beams'
        self.image_topic = '/left/image_color/compressed'

    @classmethod
    def __encode_data(cls, np_list_data: List[numpy.ndarray]) -> bytes:
        def type_encode(data_dtype: numpy.dtype):
            if data_dtype == numpy.uint8:
                return 1
            elif data_dtype == numpy.float16:
                return 2
            elif data_dtype == numpy.float32:
                return 3
            elif data_dtype == numpy.float64:
                return 4
            else:
                logging.log(logging.ERROR, "Match data type error !")
                raise ValueError
        bytes_out = b""
        for np_data in np_list_data:
            shape = np_data.shape
            shape_dim = len(shape)
            count = 1
            for shape_i in shape:
                count *= shape_i
            bytes_out += struct.pack(">h", type_encode(np_data.dtype))
            bytes_out += struct.pack(">II", count, shape_dim)
            bytes_out += struct.pack(">" + "I" * shape_dim, *shape)
            bytes_out += np_data.tobytes()
        return bytes_out

    @classmethod
    def __process_odoms(cls, odoms, lasers):
        vectors, actions, rewards = [], [], []

        final_pose = odoms[-1].pose.pose
        final_position = numpy.array([final_pose.position.x, final_pose.position.y, final_pose.position.z])
        final_quaternion = numpy.array([final_pose.orientation.x, final_pose.orientation.y, final_pose.orientation.z, final_pose.orientation.w])
        final_tf_matrix = quaternion_matrix(final_quaternion)
        random_change_yaw = random.random() *2*math.pi
        last_v, last_w = 0.0, 0.0
        min_dist = math.inf
        closer = False
        for i in range(len(odoms)):
            msg = odoms[i]
            current_pose = msg.pose.pose
            current_position = numpy.array([current_pose.position.x, current_pose.position.y, current_pose.position.z])
            current_quaternion = numpy.array([current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w])
            current_tf_matrix = quaternion_matrix(current_quaternion)
            relative_tf_matrix = numpy.dot(numpy.linalg.inv(final_tf_matrix), current_tf_matrix)
            yaw = euler_from_matrix(relative_tf_matrix)[2]
            yaw = (yaw + random_change_yaw) % (2*math.pi)
            xyz = numpy.dot(numpy.linalg.inv(current_tf_matrix[0:3, 0:3]), final_position - current_position)
            vectors.append(numpy.array([[xyz[0], xyz[1], yaw, last_v, last_w]]))
            last_v, last_w = msg.twist.twist.linear.x / 2, msg.twist.twist.angular.z / 2
            if i > 0:
                actions.append(numpy.array([[last_v, last_w]]))
                curr_dist = math.sqrt(xyz[0]**2+xyz[1]**2)
                if curr_dist < min_dist:
                    reward  = 400.0 * (curr_dist - min_dist)**2 + (1000.0 if i == len(odoms)-1 else 0.0)
                    closer = True
                else:
                    if closer == True:
                        reward = -25.0
                    closer = False
                dist_to_obs = min(list(numpy.clip(lasers[i].ranges[450:1800-450+1], 0.0, 10.0).reshape(-1)))
                if dist_to_obs < 0.5+0.17:
                    reward += -100.0 * (0.5+0.17 - dist_to_obs)**2
                rewards.append(numpy.array([[reward]]))
            min_dist = min(math.sqrt(xyz[0]**2+xyz[1]**2), min_dist)
        actions.append(numpy.zeros_like(actions[-1]))
        rewards.append(numpy.zeros_like(rewards[-1]))

        return vectors, actions, rewards

    @classmethod
    def __process_lasers(cls, scans):
        lasers = []
        for msg in scans:
            laser = numpy.array(msg.ranges[450:1800-450+1])
            laser = numpy.clip(laser, 0.0, 10.0).reshape((1, 1, 901))
            lasers.append(laser)
        return lasers

    def __exp_from_bag(self):
        bag_files = [f for f in os.listdir(self.bag_dir) if os.path.isfile(os.path.join(self.bag_dir, f))]
        if len(bag_files) < 1:
            print("ERROR: Can not find any bag.")
            raise ValueError
        bag_sizes = [os.path.getsize(os.path.join(self.bag_dir, f)) for f in bag_files]
        total_size = sum(bag_sizes)
        bag = random.choices(bag_files, [float(size)/total_size for size in bag_sizes])[0]
        bag = rosbag.Bag(os.path.join(self.bag_dir, bag))

        start_time, end_time = bag.get_start_time(), bag.get_end_time()
        begin = start_time + max(random.random() * (end_time-start_time-self.time), 0.0)
        end = min(begin+self.time, end_time)
        curr_time = begin
        curr_odom, curr_laser, curr_image = None, None, None
        odoms, lasers, images = [], [], []
        for topic, msg, t in bag.read_messages(topics=[self.odom_topic, self.laser_topic, self.image_topic], start_time=Time.from_sec(begin), end_time=Time.from_sec(end)):
            if topic == self.odom_topic:
                curr_odom = msg
            elif topic == self.laser_topic:
                curr_laser = msg
            elif topic == self.image_topic:
                curr_image = msg
            if (t.to_sec() - curr_time) >= self.dt:
                if curr_odom != None and curr_laser != None and curr_image != None:
                    odoms.append(curr_odom)
                    lasers.append(curr_laser)
                    images.append(curr_image)
                curr_time = t.to_sec()

        if len(odoms) < 1: return None, False

        vectors, actions, rewards = BagProcess.__process_odoms(odoms, lasers)
        lasers = BagProcess.__process_lasers(lasers)

        return {'laser':lasers, 'vector':vectors, 'action':actions, 'reward':rewards}, True
        
    def run(self):
        num = 0
        while num < self.sum:
            exp, bool = self.__exp_from_bag()
            if bool:
                self.collection.insert_one({'_id' : int(self.node_id*self.sum+num+self.id_bias),
                                            'laser' : BagProcess.__encode_data(exp['laser']),
                                            'vector' : BagProcess.__encode_data(exp['vector']),
                                            'action' : BagProcess.__encode_data(exp['action']),
                                            'reward' : BagProcess.__encode_data(exp['reward'])
                                            })
                num += 1
        self.client.close()
        self.database = self.collection = None
        print("[{}]Node {} Finish Exp Generate .".format(self.node_id, time.strftime('%H:%M:%S',time.localtime(time.time()))), flush=True)
