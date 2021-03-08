# ********************************************************************
# e.g. Visulization of MuJoCo when loading policy trained by TD3 AUMC.
CUDA_VISIBLE_DEVICES=0 python render_bootstrapped.py \
    --policy "TD3_aumc" \
    --env "Ant-v2" \
    --load_model "default" \
    --seed 2
