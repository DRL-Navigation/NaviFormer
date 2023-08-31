## 跑真车 Quick Start

### Compile

```
cd USTC_lab/env/drlnav_env
catkin_make --only-pkg-with-deps scan_img
```


修改 `sh/machines/all.sh`
```
ENV_FILES="
real/turtlebot.sh
"
# 注意这里turtlebot.sh里cfg的文件也需要自行配置修改
```
修改 `USTC_lab/agent/agent.py`
```
# class Agents(Process) 改成 Thread
class Agents(Thread):
```

修改 `USTC_lab/config/base_config.py`
```
# TIME_MAX = 256 改成
TIME_MAX = 256000
```

### Run
```
### 窗口1
如果模型不需要scan_img，只需要裸激光的话，窗口1可以跳过
source USTC_lab/env/drlnav_env/devel/setup.bash
roslaunch scan_img turtlebot.launch
### 窗口2
cd sh/
bash start_redis.sh
# 启动
bash start.sh
# 关闭
bash easy_kill_all.sh
```
