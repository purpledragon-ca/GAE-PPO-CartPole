# PPO with GAE for CartPole-v1 (PyTorch)

This repository implements **Proximal Policy Optimization (PPO)** using **Generalized Advantage Estimation (GAE)** in **PyTorch**, tested on the classic `CartPole-v1` environment from OpenAI Gym.


---

## Features

-  PPO-Clip Algorithm
-  Generalized Advantage Estimation (GAE)
-  Real-time rendering (via `render_mode="human"`)
-  Modify code, so it could be run on Gym ≥ v0.26.
-  Consistent Use of with torch.no_grad()

---

##  Quick Start
-  You need to have torch!
```bash
pip install gym matplotlib numpy
python main.py
```
---
##  Acknowledgement

This implementation is modified from the code at:
https://blog.csdn.net/weixin_43336108/article/details/132401350

---
## TODO
-  Multi-env parallel sampling

-  Entropy annealing

-  LSTM support for partially observable settings