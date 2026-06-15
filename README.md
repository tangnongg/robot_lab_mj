# robot lab mj

train:  
train Mjlab-RpyBhTrack-Flat-Unitree-B2 --env.scene.num-envs 4096   --agent.max-iterations 800  
train Mjlab-HoST-Ground-Unitree-G1     --env.scene.num-envs 4096   --agent.max-iterations 2000  
train Mjlab-SafeFall-G1                --env.scene.num-envs 4096   --agent.max-iterations 2000  
train Mjlab-Velocity-Flat-Unitree-G1   --env.scene.num-envs 4096   --agent.max-iterations 2000

play:  
play Mjlab-RpyBhTrack-Flat-Unitree-B2    --checkpoint_file  logs/rsl_rl/unitree_b2_rpy_bh_track/2026-06-08_23-09-54/model_799.pt  --viewer viser  
play Mjlab-HoST-Ground-Unitree-G1        --checkpoint_file  logs/rsl_rl/unitree_b2_rpy_bh_track/2026-06-08_23-09-54/model_799.pt  --viewer viser  
play Mjlab-HoST-Platform-Unitree-G1      --checkpoint_file  logs/rsl_rl/g1_host_platform/2026-06-10_17-55-14_v0_2/model_1999.pt   --viewer viser  
play Mjlab-SafeFall-G1     --checkpoint_file  logs/rsl_rl/safefall_g1/2026-06-10_21-25-46/model_1999.pt   --viewer viser  
play  Mjlab-Velocity-Flat-Unitree-G1   --checkpoint_file  logs/rsl_rl/g1_velocity/2026-06-11_01-03-02/model_1999.pt   --viewer viser
