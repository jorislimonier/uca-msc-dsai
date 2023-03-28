helpFunction()
{
   echo ""
   echo "Usage: $0 -s seed -d device"
   echo -e "\t-s seed to be used for experiment"
   echo -e "\t-d device to be used, e.g., cuda or cpu"
   exit 1 # Exit script after printing help
}

while getopts "s:d:" opt
do
   case "$opt" in
      s ) seed="$OPTARG" ;;
      d ) device="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done


# Print helpFunction in case parameters are empty
if [ -z "${seed}" ] || [ -z "${device}" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi


cd ../../


N_ROUNDS=10
LOCAL_STEPS=2
LOG_FREQ=1


echo "==> Run experiment with seed=${seed}"

python run_experiment.py \
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
  --device "${device}" \
  --log_freq "${LOG_FREQ}" \
  --verbose 1 \
  --logs_dir "logs/mnist/seed_${seed}" \
  --seed "${seed}"

