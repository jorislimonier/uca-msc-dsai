cd ../../

DEVICE="cpu"

N_ROUNDS=10
LOCAL_STEPS=2
LOG_FREQ=1

echo "Experiment with mnist dataset"

echo "=> Generate data.."

cd data/ || exit

rm -r mnist

python main.py \
  --dataset mnist \
  --n_tasks 24 \
  --iid \
  --save_dir mnist \
  --seed 1234

cd ../


echo "==> Run experiment"

for seed in 12
do
  for lr in 0.003
  do
    for server_lr in 0.1
    do
      echo "=> experiment=FedAvg | lr=${lr} | server_lr=${server_lr} | seed=${seed}"
      python run_experiment.py \
        --experiment "mnist" \
        --cfg_file_path data/mnist/cfg.json \
        --objective_type weighted \
        --aggregator_type centralized \
        --n_rounds "${N_ROUNDS}" \
        --local_steps "${LOCAL_STEPS}" \
        --local_optimizer sgd \
        --local_lr "${lr}" \
        --server_optimizer sgd \
        --server_lr "${server_lr}" \
        --train_bz 64 \
        --test_bz 1024 \
        --device "${DEVICE}" \
        --log_freq "${LOG_FREQ}" \
        --verbose 1 \
        --logs_dir "logs_tuning/mnist/mnist_lr_${lr}_server_${server_lr}/seed_${seed}" \
        --seed "${seed}"
    done
  done
done