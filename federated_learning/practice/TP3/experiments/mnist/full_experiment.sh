cd ../../

DEVICE="cpu"
SEED=12

N_ROUNDS=50
LOCAL_STEPS=5
LOG_FREQ=2

echo "=> Generate data.."

cd data/ || exit

rm -r mnist/all_clients

python3 main.py \
  --dataset mnist \
  --frac 0.01 \
  --n_clients 10 \
  --by_labels_split \
  --n_components 5 \
  --alpha 0.5 \
  --availability_parameter 0.4 \
  --stability_parameter 0.9 \
  --save_dir mnist \
  --seed 1234

cd ../

echo

name="unbiased"
local_lr=0.01
server_lr=0.3
echo "==> Run experiment with ${name} clients sampler"
python3 run_experiment.py \
  --experiment "mnist" \
  --method "fedavg" \
  --cfg_file_path data/mnist/cfg.json \
  --objective_type weighted \
  --aggregator_type centralized \
  --clients_sampler "${name}" \
  --n_rounds "${N_ROUNDS}" \
  --local_steps "${LOCAL_STEPS}" \
  --local_optimizer sgd \
  --local_lr "${local_lr}" \
  --server_optimizer sgd \
  --server_lr "${server_lr}" \
  --train_bz 64 \
  --test_bz 1024 \
  --device "${DEVICE}" \
  --log_freq "${LOG_FREQ}" \
  --verbose 1 \
  --logs_dir "logs/mnist/activity_${name}/seed_${SEED}" \
  --history_path "history/mnist/activity_${name}/seed_${SEED}.json" \
  --seed "${SEED}"


#echo

#name="biased"
#local_lr=0.01
#server_lr=3.0
#echo "==> Run experiment with ${name} clients sampler"
#python3 run_experiment.py \
#  --experiment "mnist" \
#  --method "fedavg" \
#  --cfg_file_path data/mnist/cfg.json \
#  --objective_type weighted \
#  --aggregator_type centralized \
#  --clients_sampler "${name}" \
#  --n_rounds "${N_ROUNDS}" \
#  --local_steps "${LOCAL_STEPS}" \
#  --local_optimizer sgd \
#  --local_lr "${local_lr}" \
#  --server_optimizer sgd \
#  --server_lr "${server_lr}" \
#  --train_bz 64 \
#  --test_bz 1024 \
#  --device "${DEVICE}" \
#  --log_freq "${LOG_FREQ}" \
#  --verbose 1 \
#  --logs_dir "logs/mnist/activity_${name}/seed_${SEED}" \
#  --history_path "history/mnist/activity_${name}/seed_${SEED}.json" \
#  --seed "${SEED}"


#echo

#name="perfedavg"
#local_lr=0.01
#server_lr=0.3
#echo "==> Run experiment with ${name} algorithm"
#python3 run_experiment.py \
#  --experiment "mnist" \
#  --method "${name}" \
#  --cfg_file_path data/mnist/cfg.json \
#  --objective_type weighted \
#  --aggregator_type centralized \
#  --clients_sampler "unbiased" \
#  --n_rounds "${N_ROUNDS}" \
#  --local_steps "${LOCAL_STEPS}" \
#  --local_optimizer sgd \
#  --local_lr "${local_lr}" \
#  --server_optimizer sgd \
#  --server_lr "${server_lr}" \
#  --train_bz 64 \
#  --test_bz 1024 \
#  --device "${DEVICE}" \
#  --log_freq "${LOG_FREQ}" \
#  --verbose 1 \
#  --logs_dir "logs/mnist/activity_${name}/seed_${SEED}" \
#  --history_path "history/mnist/activity_${name}/seed_${SEED}.json" \
#  --seed "${SEED}"