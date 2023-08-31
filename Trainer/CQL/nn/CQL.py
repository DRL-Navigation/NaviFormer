from nn.prenet import *
from copy import deepcopy
import math

class CQL(nn. Module):
    def __init__(self, config):
        self.discount = 0.99
        self.lr_actor = 1e-5
        self.lr_critic = 1e-3
        self.lr_alpha = 1e-3
        self.soft_target_update_rate = 5e-3
        self.cql_sample_num = 10
        self.cql_min_q_weight = 5.0

        super(CQL, self).__init__()
        self.actor = PolicyNetwork(config.n_embd)
        self.critic1 = QFuncNetwork(config.n_embd, config.n_embd)
        self.target_critic1 = deepcopy(self.critic1)
        self.critic2 = QFuncNetwork(config.n_embd, config.n_embd)
        self.target_critic2 = deepcopy(self.critic2)
        self.log_alpha = Scalar(0.0)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(list(self.critic1.parameters())+list(self.critic2.parameters()), lr=self.lr_critic)
        self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=self.lr_alpha)

        self.update_target_network(1.0)

    def update_target_network(self, soft_target_update_rate):
        def soft_target_update(network, target_network, soft_target_update_rate):
            target_network_params = {k: v for k, v in target_network.named_parameters()}
            for k, v in network.named_parameters():
                target_network_params[k].data = (
                    (1 - soft_target_update_rate) * target_network_params[k].data
                    + soft_target_update_rate * v.data
                )
        soft_target_update(self.critic1, self.target_critic1, soft_target_update_rate)
        soft_target_update(self.critic2, self.target_critic2, soft_target_update_rate)

    def forward(self, state, deterministic):
        return self.actor(state, deterministic)

    def learn(self, laser, vector, action, reward, next_laser, next_vector):
        state, next_state = [laser, vector], [next_laser, next_vector]

        new_action, new_log_pi = self.actor(state)
        q_new_actions = torch.min(
            self.critic1(state, new_action),
            self.critic2(state, new_action)
        )

        alpha_loss = -(self.log_alpha() * new_log_pi.detach()).mean()
        alpha = self.log_alpha().exp()

        policy_loss = torch.mean(alpha*new_log_pi-q_new_actions)

        q1_pred = self.critic1(state, action)
        q2_pred = self.critic2(state, action)
        next_action, next_log_pi = self.actor(next_state)
        q_target = torch.min(
            self.target_critic1(next_state, next_action),
            self.target_critic2(next_state, next_action)
        )
        td_target = reward + self.discount*q_target
        q1_loss = F.mse_loss(q1_pred, td_target.detach())
        q2_loss = F.mse_loss(q2_pred, td_target.detach())

        batch = action.shape[0]
        cql_random_actions = torch.randint(0, 11*19, size=(batch, self.cql_sample_num), dtype=torch.int64, requires_grad=False).cuda()
        cql_random_log_pis = (torch.ones(size=(batch, self.cql_sample_num, 1), dtype=torch.float32, requires_grad=False) * math.log(1.0/11/19)).cuda()
        cql_new_actions, cql_new_log_pis = self.actor.repeat_sample(state, self.cql_sample_num)
        cql_next_actions, cql_next_log_pis = self.actor.repeat_sample(next_state, self.cql_sample_num)
        cql_new_actions, cql_new_log_pis = cql_new_actions.detach(), cql_new_log_pis.detach()
        cql_next_actions, cql_next_log_pis = cql_next_actions.detach(), cql_next_log_pis.detach()
        cql_q1_random = self.critic1.multi_action_count(state, cql_random_actions)
        cql_q2_random = self.critic2.multi_action_count(state, cql_random_actions)
        cql_q1_new = self.critic1.multi_action_count(state, cql_new_actions)
        cql_q2_new = self.critic2.multi_action_count(state, cql_new_actions)
        cql_q1_next = self.critic1.multi_action_count(next_state, cql_next_actions)
        cql_q2_next = self.critic2.multi_action_count(next_state, cql_next_actions)
        cql_cat_q1 = torch.cat([cql_q1_random-cql_random_log_pis, cql_q1_next-cql_next_log_pis, cql_q1_new-cql_new_log_pis], dim=1)
        cql_cat_q2 = torch.cat([cql_q2_random-cql_random_log_pis, cql_q2_next-cql_next_log_pis, cql_q2_new-cql_new_log_pis], dim=1)
        cql_q1_ood = torch.logsumexp(cql_cat_q1, dim=1)
        cql_q2_ood = torch.logsumexp(cql_cat_q2, dim=1)
        cql_q1_diff = torch.mean(cql_q1_ood - q1_pred)
        cql_q2_diff = torch.mean(cql_q2_ood - q1_pred)
        cql_q1_loss = cql_q1_diff * self.cql_min_q_weight
        cql_q2_loss = cql_q2_diff * self.cql_min_q_weight

        q_loss = q1_loss + q2_loss + cql_q1_loss + cql_q2_loss

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()
        self.update_target_network(self.soft_target_update_rate)

        return q_loss, policy_loss

    def pred_action(self, state):
        action, _ = self.actor(state, True)
        return action
