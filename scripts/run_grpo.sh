export MASTER_PORT=10000
export NPROC_PER_NODE="8"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

CONFIG=${1:-configs/default.yaml}
# 从 config 里读出 save_root，日志写到同一目录，保证配置与产物一致
save_root=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['save_root'])" "${CONFIG}")
echo "save root: " ${save_root}
mkdir -p ${save_root}
time=$(date +%Y%m%d%H%M%S)

torchrun --nproc_per_node=${NPROC_PER_NODE} \
    main_grpo.py --config ${CONFIG} 2>&1 | tee ${save_root}/log_${time}.txt
