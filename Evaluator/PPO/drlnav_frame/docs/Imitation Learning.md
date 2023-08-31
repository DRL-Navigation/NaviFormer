## Imitation Learning

I have read several papers about IL,  almost all of them are focus on leaning on reality. However, my drlnav frame pay more attention to sim env.



### Motivation

Suppose we have trained some wonderful model, and I want to train a new model which inputs and outputs are some as original model , using Imitation learning can extremely speed up this process. 



And there are many interesting GAIL paper in DRL domain, so I think it's necessary to support GAIL in frame.



### Approach

**DAgger**:

- start with  supervise learning, we try to save lots of pairs of (States, action) generating by **Demonstrator Model**,  and train our DRL model before agents interacting with environment.
- after supervised learning, student play games while only record states in one trajectory, and teacher(demonstrator) labels action manually. This way can reduce compounding errors.  My idea is that we do not need to label manually but label by **Demonstrator Model**.

**GAIL**:

- as GAN does, there exists a discriminator to give your trajectory a score with model D, D always try to  give trajectory of expert a high score while generator's from model G low score .And our aim is to train  D, G simultaneously to make generator similar to expert.
- we use model D to give a score, as a additional reward in TRPO|PPO|A2C...
- generator just means actor.







### Reference

CS294-112 