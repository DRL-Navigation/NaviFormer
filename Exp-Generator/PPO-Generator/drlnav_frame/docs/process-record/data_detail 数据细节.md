### 数据细节

agent 传 state 到 foward， forward 返回的数据：

1. predictions  [all_num, 2] 第一列代表action_index, 第二列代表log_p
2. values, 
3. if rnd, rnd_intrinsic_rewards, **注意**：这里的reward是前一个step的reward，原因是rnd计算reward的方式是用next_states.
4. if gail, discriminatior_rewards



values第一列一定是探索的value，如果有rnd，则第二列是rnd的value，如果有gail，则最后一列是gail的value

