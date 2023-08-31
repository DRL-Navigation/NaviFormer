import random, copy, struct, pymongo, torch, math
import numpy as np
from torch.utils.data import Dataset

class NaviDataset(Dataset):
    def __init__(self, host, database='DataBase', collection='Collection'):
        super(NaviDataset, self).__init__()
        self.client = pymongo.MongoClient(host)
        self.database = self.client[database]
        self.collection = self.database[collection]
        self.Len = self.collection.estimated_document_count()

    @classmethod
    def __decode_data(cls, bytes_data):
        type_dict = {
            1 : (1, np.uint8),
            2 : (2, np.float16),
            3 : (4, np.float32),
            4 : (8, np.float64),
        }
        index, length = 0, len(bytes_data)
        list_np_data = []
        while index < length:
            c_type = struct.unpack(">h", bytes_data[index: index+2])[0]
            tb, t = type_dict[c_type]
            count, shape_dim = struct.unpack(">II", bytes_data[index+2: index+10])
            shape = struct.unpack(">" + "I" * shape_dim,
                                  bytes_data[index + 10: index + 10 + shape_dim * 4])
            index = index + 10 + shape_dim * 4
            each_bytes_data = bytes_data[index: index + count*tb]
            list_np_data.append(np.frombuffer(each_bytes_data, dtype=t).reshape(*shape))
            index += count*tb
        return list_np_data
    
    @classmethod
    def __encode_data(cls, np_list_data):
        def type_encode(data_dtype: np.dtype):
            if data_dtype == np.uint8:
                return 1
            elif data_dtype == np.float16:
                return 2
            elif data_dtype == np.float32:
                return 3
            elif data_dtype == np.float64:
                return 4
            else:
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
    def __vector_to_location(cls, vector):
        a = vector[0][0]
        b = vector[0][1]
        yaw = vector[0][2]
        x = -a*math.cos(yaw)-b*math.sin(yaw)
        y = a*math.sin(yaw)-b*math.cos(yaw)
        return np.array([[x, y, (-yaw)%(2*math.pi)]])

    @classmethod
    def __location_to_pathindex(cls, location_list):

        def distance(location1, location2):
            return math.sqrt((location1[0]-location2[0])**2+(location1[1]-location2[1])**2)
        
        def angle_from_negative_pi_to_pi(angle):
            if angle > math.pi:
                angle = angle - 2*math.pi*math.ceil(angle/(2*math.pi))
            elif angle < -math.pi:
                angle = angle - 2*math.pi*math.floor(angle/(2*math.pi))
            return angle

        def index_count(input, _min, _max, _resolution):
            input = min(_max, max(_min, input)) - _min
            return math.floor(input/_resolution)

        pathindex_list = [9]
        for i in range(1, len(location_list)):
            l = location_list[i-1][0]
            next_l = location_list[i][0]
            r = distance(next_l, l)
            t = angle_from_negative_pi_to_pi(next_l[2]-l[2]) / 2 
            r_index = index_count(r, 0.0-0.01, 1.0*0.25, 0.025)
            t_index = index_count(t, -0.9*0.25*0.5-0.005, 0.9*0.25*0.5, 0.0125)
            index = r_index*19 + t_index
            pathindex_list.append(index)
        return pathindex_list

    @classmethod
    def __data_to_exp(cls, data):
        def np_multi_add(np_list):
            if len(np_list) > 0:
                np_sum = np.zeros_like(np_list[0])
                for array in np_list:
                    np_sum = np.add(np_sum, array)
                return np_sum
            return None

        exp = {'laser':[], 'vector':[], 'reward':[]}
        for key in exp.keys():
            exp[key] = NaviDataset.__decode_data(data[key])
        length = len(exp['laser'])
        exp['location'] = list(map(NaviDataset.__vector_to_location, exp['vector']))
        pathindex = NaviDataset.__location_to_pathindex(exp['location'])
        exp['path'] = []
        for i in range(length):
            path_i = pathindex[i+1:i+5]
            path_i.extend([pathindex[-1]]*(4-len(path_i)))
            exp['path'].append(np.array(path_i).reshape((1, 4)))
        exp['return_to_go'] = copy.deepcopy(exp['reward'])
        for i in range(length-1):
            exp['return_to_go'][i] = np_multi_add(exp['reward'][i:i+4])

        return exp

    def __len__(self):
        return self.Len
    
    def __getitem__(self, index):
        return NaviDataset.__data_to_exp(self.collection.find_one({'_id': int(index)}))
        # return NaviDataset.__data_to_exp(self.collection.aggregate([{'$sample': {'size': 1}}]).next())

class NaviCollateFN:
    def __init__(self, max_len):
        self.max_len = max_len

    def collate_fn(self, batch):
        exp_batch = {'laser':[], 'location':[], 'path':[], 'return_to_go':[]}
        mask = []

        for exp in batch:
            elen = len(exp['laser'])
            if elen < 2: continue
            for _ in range(4):
                ti = random.randint(1, self.max_len)
                si = random.randint(0, elen-2)
                tlen = ti if si+ti < elen else elen-si-1
                if tlen < 1: continue
                for key in exp_batch.keys():
                    exp_batch[key].append(np.concatenate([exp[key][si+j] for j in range(tlen)], axis=0))
                    exp_batch[key][-1] = np.concatenate([np.zeros((self.max_len-tlen,)+exp_batch[key][-1].shape[1:]), exp_batch[key][-1]], axis=0)
                    exp_batch[key][-1] = exp_batch[key][-1].reshape((1,)+exp_batch[key][-1].shape)
                mask.append(np.concatenate([np.zeros((1, self.max_len-tlen)), np.ones((1, tlen))], axis=1))

        for key in exp_batch.keys():
            exp_batch[key] = torch.from_numpy(np.concatenate(exp_batch[key], axis=0)).to(dtype=torch.float32)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(dtype=torch.bool)

        return exp_batch['location'], exp_batch['laser'], exp_batch['path'], exp_batch['return_to_go'], mask