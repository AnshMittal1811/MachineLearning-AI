# soccer_juggle_release


Code to train a virtual character to juggle a soccer ball with reinforcement learning. Install my fork of [RAISIM](https://github.com/ZhaomingXie/raisimLib/tree/juggling) 
first and put the this project inside raisimGymTorch/raisimGymTorch/env/envs/.

vec_ppo.py allows you to train soccer juggling from scratch. It should take around 30000 iterations to obtain a good policy.
Depends on your hardware setup, this could take between 12-30 hours.

Run test_policy.py to visualize a pretrain policy.

ActorCrititNetMann in model.py implements our layer-wise Mixture of Experts, which is mathematically equivalent to the weight-blended Mixture of Expert
introduced in the paper [Mode-Adaptive Neural Networks for Quadruped Motion Control](https://homepages.inf.ed.ac.uk/tkomura/dog.pdf). Our weight-blended
interpretation allows for up to 10x speed up compared to the original implementation. This implementation is inspired by [Motion VAE](https://github.com/electronicarts/character-motion-vaes).
