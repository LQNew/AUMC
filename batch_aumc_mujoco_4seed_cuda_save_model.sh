#!/bin/bash
ENV=$1
policy=$2
CudaNum=$3
BETA=$4

# Script to reproduce results
for ((i=24;i<28;i+=4))
do 
	CUDA_VISIBLE_DEVICES=$CudaNum python main_aumc.py \
		--policy ${policy} \
		--env $ENV \
		--beta $BETA \
		--save_model \
		--seed $i \
		--exp_name "${policy}-${ENV}" & \
	CUDA_VISIBLE_DEVICES=$CudaNum python main_aumc.py \
		--policy ${policy} \
		--env $ENV \
		--beta $BETA \
		--save_model \
		--seed $[$i+1] \
		--exp_name "${policy}-${ENV}" & \
	CUDA_VISIBLE_DEVICES=$CudaNum python main_aumc.py \
		--policy ${policy} \
		--env $ENV \
		--beta $BETA \
		--save_model \
		--seed $[$i+2] \
		--exp_name "${policy}-${ENV}" & \
	CUDA_VISIBLE_DEVICES=$CudaNum python main_aumc.py \
		--policy ${policy} \
		--env $ENV \
		--beta $BETA \
		--save_model \
		--seed $[$i+3] \
		--exp_name "${policy}-${ENV}"
done
