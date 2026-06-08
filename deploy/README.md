# Deploy

实物部署前先确保主机安装了下列通信工具：  

- [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git)

- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git)

```bash
# Before deployment, install the required communication tools:
# cyclonedds
# unitree_sdk2
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics ..   # 指定/opt/unitree_robotics重要！默认是/usr/local/
# modern method:
cmake --build .
cmake --build . --target install

# old method：
make ..
make install
```

```bash
sudo apt update
sudo apt install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
```

## 仿真部署

**启动仿真器(注意此处需连接上手柄才能启动)：**

在实物部署前，建议使用[unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)进行仿真部署，防止实物机器人出现异常动作。本框架已将其集成。

编译unitree_mujoco：

```bash
cd simulate
mkdir build && cd build
cmake .. && make -j8 2>&1 | tee build.log
```

执行 ：

```bash
./simulate/build/unitree_mujoco
```

**仿真控制程序：**

以 Unitree b2 速度控制为例（其他机器人同理）

在 `simulate/config` 中选择对应机器人  

将策略文件（`policy.onnx`）放入`deploy/robots/g1/config/policy/velocity/vo/exported` 下，然后执行：

```bash
cd deploy/robots/b2
mkdir build && cd build
cmake .. && make
```

重新build并执行：

```bash
cd .. && rm -rf build
mkdir build && cd build
cmake .. && make -j8
./b2_ctrl --network=lo
```


