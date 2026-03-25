#!/bin/bash
set -e
cd /teamspace/studios/this_studio/PCGT
source venv/bin/activate
echo "[$(date +%H:%M:%S)] Convergence fix start"

cd medium
for DS in chameleon squirrel; do
  for METHOD in sgformer pcgt; do
    echo "[$(date +%H:%M:%S)] $DS/$METHOD"
    EXTRA=""
    K_ARG=""
    if [ "$METHOD" = "pcgt" ]; then K_ARG="--num_partitions 10"; fi
    if [ "$DS" = "squirrel" ] && [ "$METHOD" = "pcgt" ]; then EXTRA="--num_layers 4"; fi
    python main.py --dataset "$DS" --method "$METHOD" --data_dir ../data \
      --hidden_channels 64 --lr 0.01 --weight_decay 5e-4 \
      --dropout 0.5 --epochs 500 --runs 1 --display_step 1 \
      --graph_weight 0.8 --aggregate add $K_ARG $EXTRA --device 0 \
      > "../logs/convergence_${DS}_${METHOD}.log" 2>&1
    echo "[$(date +%H:%M:%S)] done $DS/$METHOD"
  done
done
cd ..

# Then run arxiv 3-run verification
echo "[$(date +%H:%M:%S)] arxiv PCGT 3-run"
cd large
python main.py --dataset ogbn-arxiv --method pcgt \
  --hidden_channels 256 --lr 0.001 --epochs 500 --runs 3 \
  --gnn_num_layers 3 --trans_num_layers 1 --trans_dropout 0.5 \
  --gnn_dropout 0.5 --use_graph --aggregate add --batch_size 10000 \
  --num_partitions 256 --device 0 > "../logs/arxiv_pcgt_3run.log" 2>&1
cd ..
echo "[$(date +%H:%M:%S)] ALL DONE"
