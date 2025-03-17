# UAV-Stability-improve-method
Project for enhancing UAV Stability by using reinforcement learning method. Including test DDPG algorithms on MountainCarContinuous environment and Environment Building for UAV base on matlab UAV model. 尝试使用分阶段强化学习：1. 起飞阶段 2. 飞行途中控制阶段 3.降落阶段


## TODO
### 环境测试
#### TODO:
需要进行归一化操作//
期望位置：des_x1, des_y1, des_z1；实际位置 act_x1, act_y1, act_z1 ：两者进行计算后作为一个输入给agent 或者计算当前位置和waypoints的差值: diff_pos//
输入量：diff_pos, propeller, qc_motor, limit_altitude, limit_attitude, limit_motor, kp_altitude, ki_altitude, kd_altitude, kp_position, kd_position, kp_attitude, kd_attitude, kp_yaw, kd_yaw//
输出量：4个螺旋桨的cmd

### 奖励函数
#### TODO:

### 目标
#### TODO:

### 算法检验
#### TODO:
修改算法探索率:sigmma 为动态收敛的,减少产生的噪声



