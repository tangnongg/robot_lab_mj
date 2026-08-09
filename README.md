# robot lab mj

train:  
train Mjlab-RpyBhTrack-Flat-Unitree-B2 --env.scene.num-envs 4096   --agent.max-iterations 800  
train Mjlab-HoST-Ground-Unitree-G1     --env.scene.num-envs 4096   --agent.max-iterations 2000  
train Mjlab-SafeFall-G1                --env.scene.num-envs 4096   --agent.max-iterations 2000  
train Mjlab-Velocity-Flat-Unitree-G1   --env.scene.num-envs 4096   --agent.max-iterations 2000

play:  
play Mjlab-RpyBhTrack-Flat-Unitree-B2    --checkpoint_file  logs/rsl_rl/unitree_b2_rpy_bh_track/2026-06-08_23-09-54/model_799.pt  --viewer viser  
play Mjlab-HoST-Ground-Unitree-G1        --checkpoint_file  logs/rsl_rl/g1_host_ground/2026-07-28_22-54-13_supine_v3_latched/model_799.pt  --viewer viser
play Mjlab-HoST-Platform-Unitree-G1      --checkpoint_file  logs/rsl_rl/g1_host_platform/2026-06-10_17-55-14_v0_2/model_1999.pt   --viewer viser  
play Mjlab-SafeFall-G1     --checkpoint_file  logs/rsl_rl/safefall_g1/2026-06-10_21-25-46/model_1999.pt   --viewer viser  
play  Mjlab-Velocity-Flat-Unitree-G1   --checkpoint_file  logs/rsl_rl/g1_velocity/2026-06-11_01-03-02/model_1999.pt   --viewer viser

evaluate：
CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl WARP_CACHE_PATH=/tmp/robot_lab_warp_gpu3 MPLCONFIGDIR=/tmp/robot_lab_mpl_gpu3 PYTHONPATH=src:/data1/tanglong/mjlab/src /data1/tanglong/miniconda3/envs/env_mjlab_e/bin/python -u -m robot_lab.tasks.host.record_video --checkpoint logs/rsl_rl/g1_host_ground/2026-08-09_12-15-25_host_multicritic_ground_v1_to1000/model_1000.pt --frames 500 --device cuda:0 --force 0.0 --action-rescale 0.25 --output logs/rsl_rl/g1_host_ground/2026-08-09_12-15-25_host_multicritic_ground_v1_to1000/videos/model_1000.mp4
