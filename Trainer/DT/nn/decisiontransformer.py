import transformers
from nn.prenet import *
from torch.cuda.amp import autocast


class DT(nn.Module):
    def __init__(self, config:transformers.GPT2Config, checkpoint:bool=False):
        super(DT, self).__init__()
        self.checkpoint = checkpoint
        self.token_dim = config.n_embd
        config.use_cache = not checkpoint
        self.GPT = transformers.GPT2Model(config)
        self.GPT.gradient_checkpointing = self.checkpoint
        self.state_embed = StateEmbed(self.token_dim)
        self.rtg_embed = RTGEmbed(self.token_dim)
        self.action_embed = DiscreteActionEmbed(dim=self.token_dim)
        self.layernorm = nn.LayerNorm(self.token_dim)
        self.action_predict = DiscreteActionPredict(dim=self.token_dim)

    @autocast()
    def forward(self, laser, vector, rtg, action, mask):
        batch_size, seq_length = rtg.shape[0], rtg.shape[1]
        state_token = self.state_embed([laser, vector])
        rtg_token = self.rtg_embed(rtg)
        action_token = self.action_embed(action)
        tokens = torch.stack([rtg_token, state_token, action_token], dim=1).permute(0, 2, 1, 3).reshape(batch_size, seq_length*3, self.token_dim)
        mask = torch.stack([mask]*3, dim=1).permute(0, 2, 1).reshape(batch_size, seq_length*3)
        mask_pt = torch.argmax(mask.to(dtype=torch.int64), dim=1)
        position_ids = torch.stack([torch.cat([torch.zeros((mask_pt[i].item(),), dtype=torch.int64), torch.arange(1, mask.shape[1]-mask_pt[i].item()+1, dtype=torch.int64)], dim=0) for i in range(batch_size)], dim=0).to(dtype=torch.int64, device=mask.device, non_blocking=True)
        tokens = self.layernorm(tokens)
        output_tokens = self.GPT(inputs_embeds=tokens, attention_mask=mask, position_ids=position_ids)['last_hidden_state'].reshape(batch_size, seq_length, 3, self.token_dim).permute(0, 2, 1, 3)
        action = self.action_predict(output_tokens[:, 1:2])
        return action

    @autocast()
    def learn(self, laser, vector, rtg, action, mask):
        batch_size, seq_length = rtg.shape[0], rtg.shape[1]
        action_pred = self.forward(laser, vector, rtg, action, mask)
        action = action.reshape((batch_size*seq_length,-1))[mask.reshape(-1)==True]
        action_pred = action_pred.reshape((batch_size*seq_length,-1))[mask.reshape(-1)==True]
        loss = F.nll_loss(torch.log(action_pred), torch.argmax(action, dim=1))
        for p in self.parameters():
            loss += 0.0 * p.sum()
        return loss

    def pred_action(self, states, rtg, action):
        with torch.no_grad():
            seq_length = rtg.shape[0]
            laser = states[0].unsqueeze(0)
            vector = states[1].unsqueeze(0)
            state_token = self.state_embed([laser, vector])
            rtg_token = self.rtg_embed(rtg.unsqueeze(0))
            action_token = self.action_embed(action.unsqueeze(0))
            tokens = torch.stack([rtg_token, state_token, action_token], dim=1).permute(0, 2, 1, 3).reshape(1, seq_length*3, self.token_dim)
            mask = torch.ones((1, seq_length)).to(dtype=torch.bool, device=tokens.device)
            mask = torch.stack([mask]*3, dim=1).permute(0, 2, 1).reshape(1, seq_length*3)
            mask, _ = mask.split([seq_length*3-1, 1], dim=1)            
            mask = torch.cat([mask, torch.zeros((1, 1)).to(dtype=torch.bool, device=tokens.device)], dim=1)
            position_ids = torch.arange(1, mask.shape[1]+1, device=tokens.device, dtype=torch.int64).unsqueeze(0)
            tokens = self.layernorm(tokens)
            output_tokens = self.GPT(inputs_embeds=tokens, attention_mask=mask, position_ids=position_ids)['last_hidden_state'].reshape(1, seq_length, 3, self.token_dim).permute(0, 2, 1, 3)
            action_pred = self.action_predict(output_tokens[:, 1:2].squeeze(1))
            torch.cuda.empty_cache()
        return action_pred[0]
