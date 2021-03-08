# e.g. for training RL agent in MuJoCo Env.

# **********************************************************
# DDPG AUMC
CUDA_VISIBLE_DEVICES=0 python main_aumc.py \
    --policy "DDPG_aumc" \
    --env "Walker2d-v2" \
    --beta 0.4 \
    --seed 1 \
    --save_model \
    --exp_name DDPG_aumc-Walker2d-v2

# **********************************************************
# TD3 AUMC
CUDA_VISIBLE_DEVICES=0 python main_aumc.py \
    --policy "TD3_aumc" \
    --env "Ant-v2" \
    --beta 0.6 \
    --seed 0 \
    --save_model \
    --exp_name TD3_aumc-Ant-v2


# **********************************************************
# SAC AUMC
CUDA_VISIBLE_DEVICES=0 python main_aumc.py \
    --policy "SAC_aumc" \
    --env "Swimmer-v2" \
    --beta 0.8 \
    --seed 0 \
    --exp_name SAC_aumc-Swimmer-v2
