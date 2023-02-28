cd ../../

DEVICE="cpu"

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

cd ../experiments/mnist/

echo "Script executed from: ${PWD}"

chmod +x run.sh

for seed in 12 42 62
do
  ./run.sh -s "${seed}" -d "${DEVICE}"
done