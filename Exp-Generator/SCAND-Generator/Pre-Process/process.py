from bags_to_mongodb import BagProcess
from multiprocessing import Process

class MultiProcess(Process):
    def __init__(self, rank):
        super(MultiProcess, self).__init__()
        self.bag_dir = '../Data'
        self.sum = 2000
        self.id_bias = 0

        self.rank  = rank
        self.process = BagProcess(self.bag_dir, self.sum, self.rank, self.id_bias)

    def run(self):
        self.process.run()


process_list = []
for rank in range(50):
    p = MultiProcess(rank)
    p.start()
    process_list.append(p)

for p in process_list:
    p.join()

print("All Finish Exp Generate .")