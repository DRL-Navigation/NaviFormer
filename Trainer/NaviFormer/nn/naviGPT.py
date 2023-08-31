import transformers
from nn.prenet import *
from torch.cuda.amp import autocast


class NaviGPT(nn.Module):
    def __init__(self, config:transformers.GPT2Config, checkpoint:bool=False):
        super(NaviGPT, self).__init__()
        self.checkpoint = checkpoint
        self.token_dim = config.n_embd
        config.use_cache = not checkpoint
        self.GPT = transformers.GPT2Model(config)
        self.GPT.gradient_checkpointing = self.checkpoint
        self.state_embed = StateEmbed(self.token_dim)
        self.path_embed = PathEmbed(self.token_dim)
        self.rtg_embed = RTGEmbed(self.token_dim)
        self.layernorm = nn.LayerNorm(self.token_dim)
        self.path_predict = PathPredict(dim=self.token_dim)

    @autocast()
    def forward(self, location, laser, path, rtg, mask):
        batch_size, seq_length = rtg.shape[0], rtg.shape[1]
        state_tokens = self.state_embed(location, laser)
        rtg_token = self.rtg_embed(rtg)
        path_tokens = self.path_embed(path)
        state_len = len(state_tokens)
        path_len = len(path_tokens)
        tokens = torch.stack([rtg_token,]+state_tokens+path_tokens, dim=1).permute(0, 2, 1, 3).reshape(batch_size, seq_length*(1+state_len+path_len), self.token_dim)
        mask = torch.stack([mask]*(1+state_len+path_len), dim=1).permute(0, 2, 1).reshape(batch_size, seq_length*(1+state_len+path_len))
        mask_pt = torch.argmax(mask.to(dtype=torch.int64), dim=1)
        position_ids = torch.stack([torch.cat([torch.zeros((mask_pt[i].item(),), dtype=torch.int64), torch.arange(1, mask.shape[1]-mask_pt[i].item()+1, dtype=torch.int64)], dim=0) for i in range(batch_size)], dim=0).to(dtype=torch.int64, device=mask.device, non_blocking=True)
        tokens = self.layernorm(tokens)
        output_tokens = self.GPT(inputs_embeds=tokens, attention_mask=mask, position_ids=position_ids)['last_hidden_state'].reshape(batch_size, seq_length, (1+state_len+path_len), self.token_dim).permute(0, 2, 1, 3)
        path, softmax = self.path_predict(output_tokens[:, state_len:state_len+path_len])
        return path.permute(0, 2, 1), softmax.permute(0, 2, 1, 3)

    @autocast()
    def learn(self, location, laser, path, rtg, mask, discount=0.5):
        batch_size, seq_length, path_length = path.shape[0], path.shape[1], path.shape[2]
        _, softmax = self.forward(location, laser, path, rtg, mask)
        path = path.reshape((batch_size*seq_length,)+path.shape[2:])[mask.reshape(-1)==True].to(dtype=torch.int64)
        softmax = softmax.reshape((batch_size*seq_length,)+softmax.shape[2:])[mask.reshape(-1)==True]
        loss = None
        for i in range(path_length):
            path_i = path[:,i]
            softmax_i = softmax[:,i]
            if loss == None:
                loss = F.nll_loss(torch.log(softmax_i), path_i)*(discount**i)
            else:
                loss += F.nll_loss(torch.log(softmax_i), path_i)*(discount**i)
        for p in self.parameters():
            loss += 0.0 * p.sum()
        return loss

    def pred_path(self, location, laser, path, rtg, path_topk = [0, 0, 0, 0]):
        with torch.no_grad():
            seq_length, path_len = path.shape[0], path.shape[1]
            state_tokens = self.state_embed(location.unsqueeze(0), laser.unsqueeze(0))
            state_len = len(state_tokens)
            rtg_token = self.rtg_embed(rtg.unsqueeze(0))

            for i in range(path_len):
                path_tokens = self.path_embed(path.unsqueeze(0))
                tokens = torch.stack([rtg_token,]+state_tokens+path_tokens, dim=1).permute(0, 2, 1, 3).reshape(1, seq_length*(1+state_len+path_len), self.token_dim)
                mask = torch.ones((1, seq_length)).to(dtype=torch.bool, device=tokens.device)
                mask = torch.stack([mask]*(1+state_len+path_len), dim=1).permute(0, 2, 1).reshape(1, seq_length*(1+state_len+path_len))
                mask, _ = mask.split([seq_length*(1+state_len+path_len)-path_len+i, path_len-i], dim=1)            
                mask = torch.cat([mask, torch.zeros((1, path_len-i)).to(dtype=torch.bool, device=tokens.device)], dim=1)
                position_ids = torch.arange(1, mask.shape[1]+1, device=tokens.device, dtype=torch.int64).unsqueeze(0)
                tokens = self.layernorm(tokens)
                output_tokens = self.GPT(inputs_embeds=tokens, attention_mask=mask, position_ids=position_ids)['last_hidden_state'].reshape(1, seq_length, (1+state_len+path_len), self.token_dim).permute(0, 2, 1, 3)
                _, softmax = self.path_predict(output_tokens[:, state_len:state_len+path_len])
                softmax = softmax.permute(0, 2, 1, 3).squeeze(0)[-1][i]
                sorted_index = torch.argsort(softmax, descending=True)
                topk = path_topk[i]
                path[-1][i] = sorted_index[min(topk, sorted_index.shape[0]-1)]
            torch.cuda.empty_cache()
        return path[-1]