string name
string ktype
geometry_msgs/Pose init_pose
geometry_msgs/Point goal
string shape
float32[] size
bool alive
float32 v
float32 w
float32 v_y
float32 max_speed
string env_name
float32[] sensor_cfg
geometry_msgs/Point[] trajectory
# for ped dataset
geometry_msgs/Point[] trajectory_v

comn_pkg/SpeedLimiter speed_limiter_v
comn_pkg/SpeedLimiter speed_limiter_w