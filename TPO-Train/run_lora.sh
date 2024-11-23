srun -p compute --nodes=1 \
  -t 24:00:00 \
  --gpus-per-node=1 \
  torchrun --standalone --nnodes=1 --nproc-per-node 1 finetune.py \
  --vla_path "/data/zhaoyang_wang/projects/OpenVLA/openvla/ckpt/fullmodel" \
  --dataset_name "rlds_np_rollout" \
  --chosen_traj_dir /data/zhaoyang_wang/projects/OpenVLA/openvlatraj/suc_new \
  --rejected_traj_dir /data/zhaoyang_wang/projects/OpenVLA/openvlatraj/fail_new \
  --run_root_dir /data/zhaoyang_wang/projects/OpenVLA/openvla/model/checkpoints \
  --adapter_tmp_dir /data/zhaoyang_wang/projects/OpenVLA/openvla/ckpt \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --image_aug False \
  --wandb_project openvla-ft \
  --wandb_entity tsinghuaair \
  --save_steps 5000