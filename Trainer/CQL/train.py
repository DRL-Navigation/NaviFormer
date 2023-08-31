import torch, tqdm, os
import torch.utils.tensorboard as tensorboard
from torch.utils.data import DataLoader as DL
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from data import NaviDataset, NaviCollateFN
from nn import CQL, GPT2Config

class Train:
    def __init__(self, model, dataloader, epoch=10000, grad_norm_clip=None, fine_tune=None, name="model", rank=0, device="cuda", world_size=1):
        self.model = model
        self.dataloader = dataloader
        self.epoch = epoch
        self.grad_norm_clip = grad_norm_clip
        if fine_tune != None:
            self.model.module.load_state_dict(torch.load(fine_tune))
        self.tb_writer = None
        self.name = name
        self.rank = rank
        self.device = device
        self.world_size = world_size

    def __run(self, data, epoch)->None:
        self.model.train()
        q_loss, p_loss = self.model.learn(*data)
        if self.tb_writer != None:
            self.tb_writer.add_scalar('Q_Func_Loss', q_loss.item(), epoch)
            self.tb_writer.add_scalar('Policy_Loss', p_loss.item(), epoch)

    def Run(self)->None:
        if self.rank == 0:
            print('--Training Process Running:', flush=True)
            if not os.path.exists('./log'): os.mkdir('./log')
            if not os.path.exists('./log/{}'.format(self.name)): os.mkdir('./log/{}'.format(self.name))
            self.tb_writer = tensorboard.SummaryWriter(log_dir='./log/{}/tfboard'.format(self.name))

        data_iter = iter(self.dataloader)
        epoch_iter = range(self.epoch) if self.rank != 0 else tqdm.tqdm(range(self.epoch), '', ncols=120, unit='Epoch')
        for epoch in epoch_iter:
            data = next(data_iter, None)
            if data is None:
                data_iter = iter(self.dataloader)
                data = next(data_iter)
            for i in range(len(data)):
                data[i] = data[i].cuda(device=self.device, non_blocking=True)
            self.__run(data, epoch)
            if (self.rank == 0) and ((epoch+1) % 2000 == 0):
                torch.save(self.model.state_dict(), './log/{}/'.format(self.name)+str(int((epoch+1)/1000))+'K.pt')

        if self.rank == 0:
            torch.save(self.model.state_dict(), './log/{}/last_model.pt'.format(self.name))
            print('--Finish: Models are saved in ./log/{} .'.format(self.name), flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="model")
    parser.add_argument("--fine_tune", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="mongodb://localhost:27017/")
    parser.add_argument("--max_len", type=int, default=5)
    parser.add_argument("--token_dim", type=int, default=512)
    parser.add_argument("--nlayer", type=int, default=3)
    parser.add_argument("--nhead", type=int, default=1)
    parser.add_argument("--ninner", type=int, default=128*4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=10**-5)
    parser.add_argument("--weight_decay", type=float, default=10**-5)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=50000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--grad_norm_clip", type=float, default=None)
    parser.add_argument("--checkpoint", type=bool, default=False)
    parse = parser.parse_args()

    torch.backends.cudnn.enable=True
    torch.backends.cudnn.benchmark=True

    dataset = NaviDataset(parse.dataset)
    collate = NaviCollateFN(parse.max_len)
    dataloader = DL(dataset=dataset, batch_size=parse.batch, num_workers=1, shuffle=True, collate_fn=collate.collate_fn, drop_last=True, pin_memory=True)
    config = GPT2Config(n_embd=parse.token_dim, n_layer=parse.nlayer, n_head=parse.nhead, n_inner=parse.ninner, resid_pdrop=parse.dropout)
    model = CQL(config).cuda()
    train = Train(model, dataloader, parse.epoch, parse.grad_norm_clip, parse.fine_tune, parse.name)
    train.Run()