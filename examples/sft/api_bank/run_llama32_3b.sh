#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:4
#SBATCH --time=240:00:00
#SBATCH --job-name=sft
#SBATCH --output slurm/%j.out

cd /iris/u/rypark/code/verl/examples/sft/api_bank
source ~/.bashrc
conda activate verlc
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/iris/u/rypark/code/dense-tool-rewards/data/train.parquet \
    data.val_files=/iris/u/rypark/code/dense-tool-rewards/data/test.parquet \
    data.prompt_key=input \
    data.response_key=output \
    data.pretemplated=true \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=4096 \
    model.partial_pretrain=meta-llama/Llama-3.2-3B-Instruct \
    trainer.default_local_dir=/iris/u/rypark/code/dense-tool-rewards/models/llama_32_3b_instruct_l1l2_dedup \
    trainer.project_name=dense-tool-rewards \
    trainer.experiment_name=llama-3.2-3b-instruct-l1l2 \
    trainer.total_epochs=5 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@
