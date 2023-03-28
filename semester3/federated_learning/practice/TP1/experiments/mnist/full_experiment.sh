cd ../../

DEVICE="cuda"
SEED=12

N_ROUNDS=10
LOCAL_STEPS=2
LOG_FREQ=1

echo "=> Generate data.."

cd data/ || exit

rm -r mnist

python3 main.py \
  --dataset mnist \
  --n_tasks 24 \
  --iid \
  --save_dir mnist \
  --seed 1234

cd ../

echo "==> Run experiment"

python3 run_experiment.py \
  --experiment "mnist" \
  --cfg_file_path data/mnist/cfg.json \
  --objective_type weighted \
  --aggregator_type centralized \
  --n_rounds "${N_ROUNDS}" \
  --local_steps "${LOCAL_STEPS}" \
  --local_optimizer sgd \
  --local_lr 0.003 \
  --server_optimizer sgd \
  --server_lr 0.1 \
  --train_bz 64 \
  --test_bz 1024 \
  --device "${DEVICE}" \
  --log_freq "${LOG_FREQ}" \
  --verbose 1 \
  --logs_dir "logs/mnist/seed_${SEED}" \
  --seed "${SEED}"