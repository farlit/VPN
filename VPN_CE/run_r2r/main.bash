export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export TORCH_DISTRIBUTED_DEBUG=DETAIL

flag1="--exp_name release_r2r_pre_ord
      --run-type train
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0,1]
      TORCH_GPU_IDS [0,1]
      GPU_NUMBERS 2
      NUM_ENVIRONMENTS 8
      IL.iters 400000
      IL.lr 1e-5
      IL.log_every 1000
      IL.ml_weight 1.0
      IL.load_from_ckpt False
      IL.is_requeue True
      IL.waypoint_aug  True
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      MODEL.pretrained_path None
      "

#flag2=" --exp_name release_r2r_pre
#      --run-type eval
#      --exp-config run_r2r/iter_train.yaml
#      SIMULATOR_GPU_IDS [0,1]
#      TORCH_GPU_IDS [0,1]
#      GPU_NUMBERS 2
#      NUM_ENVIRONMENTS 8
#      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
#      EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_pre/ckpt.iter102000.pth
#      IL.back_algo control
#      MODEL.pretrained_path None
#      "

flag2=" --exp_name release_r2r_ord
      --run-type eval
      --exp-config run_r2r/iter_train.yaml
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_IDS [0]
      GPU_NUMBERS 1
      NUM_ENVIRONMENTS 8
      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
      EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_pre_ord/ckpt.iter109000.pth
      IL.back_algo control
      MODEL.pretrained_path None
      "

#flag2=" --exp_name release_r2r_pre_ord
#      --run-type eval
#      --exp-config run_r2r/iter_train.yaml
#      SIMULATOR_GPU_IDS [0]
#      TORCH_GPU_IDS [0]
#      GPU_NUMBERS 1
#      NUM_ENVIRONMENTS 8
#      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
#      EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_pre_ord
#      IL.back_algo control
#      MODEL.pretrained_path None
#      "

#flag3="--exp_name release_r2r
#      --run-type inference
#      --exp-config run_r2r/iter_train.yaml
#      SIMULATOR_GPU_IDS [0,1]
#      TORCH_GPU_IDS [0,1]
#      GPU_NUMBERS 2
#      NUM_ENVIRONMENTS 8
#      TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
#      INFERENCE.CKPT_PATH data/logs/checkpoints/release_r2r/ckpt.iter12000.pth
#      INFERENCE.PREDICTIONS_FILE preds.json
#      IL.back_algo control
#      "

mode=$1
case $mode in 
      train)
      echo "###### train mode ######"
      python -m torch.distributed.launch --nproc_per_node=2 --master_port $2 run.py $flag1
      ;;
      eval)
      echo "###### eval mode ######"
      python -m torch.distributed.launch --nproc_per_node=1 --master_port $2 run.py $flag2
      ;;
      infer)
      echo "###### infer mode ######"
      python -m torch.distributed.launch --nproc_per_node=2 --master_port $2 run.py $flag3
      ;;
esac