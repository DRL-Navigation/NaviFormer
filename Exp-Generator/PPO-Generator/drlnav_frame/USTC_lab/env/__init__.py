# -*- coding: utf-8 -*-
# @Time    : 2021/2/23 10:55 上午
# @Author  : qiuqc@mail.ustc.edu.cn
# @FileName: __init__.py

import sys
sys.path.append('drlnav_env')
from USTC_lab.env.vec_env import VecEnv, make_vecenv
from USTC_lab.env.gym_env import make_gymenv
try:
    from USTC_lab.env.drlnav_env.envs import make_env as make_robotnavenv
except Exception as e:
    print(str(e))
    print("Guess you are not clone repo with ----recurse-submodules,"
          " you can try: \ngit submodule init & git submodule update")
    make_robotnavenv = None

env_types = ['gym', 'mujoco', 'robot_nav', 'real_env', 'gazebo_env']
env_dict = {
    'gym': make_gymenv,
    "robot_nav": make_robotnavenv,
    'real_env': make_robotnavenv,
    'gazebo_env': make_robotnavenv,
}


def make_env(cfg: dict):
    env_type = cfg['env_type']
    assert env_type in env_types
    return env_dict[env_type](cfg)


__all__ = [
    'make_robotnavenv',
    "make_gymenv",
    "make_vecenv",

]
