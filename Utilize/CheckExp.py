import struct, random, copy, pymongo, time
import numpy as np
from typing import List

class DataBase:
    def __init__(self, host, database, collection):
        self.client = pymongo.MongoClient(host)
        self.database = self.client[database]
        self.collection = self.database[collection]
        self.Len = self.collection.estimated_document_count()

    @classmethod
    def __decode_data(cls, bytes_data: bytes) -> List[np.ndarray]:
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
    def __encode_data(cls, np_list_data: List[np.ndarray]) -> bytes:
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
    def __data_to_exp(cls, data):
        exp = {'laser':[], 'vector':[], 'ped':[], 'action':[], 'reward':[]}
        for key in exp.keys():
            exp[key] = DataBase.__decode_data(data[key])
        length = len(exp['reward'])
        exp['return_to_go'] = copy.deepcopy(exp['reward'])
        for j in range(2, length+1):
            exp['return_to_go'][length-j] = np.add(exp['return_to_go'][length-j], exp['return_to_go'][length-j+1])
        return exp

    def random_get(self, index=None):
        if index == None:
            index = random.randint(0, self.Len-1)
        document = self.collection.find_one({'_id': int(index)})
        return DataBase.__data_to_exp(document)

    def close(self):
        if self.client != None:
            self.client.close()
            self.client = self.database = self.collection = None

host = 'mongodb://localhost:27017/'
database = 'DataBase'
collection = 'Collection'

db = DataBase(host, database, collection)
Time = time.time()
exp = db.random_get()
Time = time.time() - Time
print('DataBase:', db.Len, '\n')
print('GetOneCost:', Time, '\n')
db.close()



for si in range(len(exp['action'])):
    print('action:', exp['action'][si], '\n')
    print('reward:', exp['reward'][si], '\n')

print('All_RTG:', exp['return_to_go'][0], '\n')
print('length:', len(exp['action']), '\n')
ti = random.randint(0, len(exp['action'])-1)
print('sample_state:\n', 'laser:', exp['laser'][ti], '\n', 'vector:', exp['vector'][ti], '\n', 'ped:', exp['ped'][ti], '\n')