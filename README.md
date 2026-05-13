# migrate "go2w loco" task of robot lab from IsaacLab to mjlab

在这个project中有两个task，anymal_c_velocity和go2w_loco,其中的anymal_c_velocity来自mjlab的example，用来说明如何组织目录结构，见Create_Task.md

**可能遇到的问题**：  
将robot_lab从isaaclab迁移到mjlab，类似isaaclab的external project，这里也在mjlab外部新建一个project folder。
如果pip install -e . 之后修改了包的结构：

```toml
# pyproject.toml:

[project.entry-points."mjlab.tasks"]
robot_lab = "robot_lab.tasks"
```

即使uninstall之后再pip install -e .，可能使得.pth文件，.dist-info目录的内容有错误（没有被更新），解决办法是删除robo_lab或者anymal_c_velocity相关package的pth文件，.dist-info目录，重新pip install -e .






contact sonsor
NaN


主要关注的文件：
asset_zoo/robot/unitree_go2w/

task/go2w/env_cfg.py
task/go2w/velocity_env_cfg.py
task/go2w/mdp/

