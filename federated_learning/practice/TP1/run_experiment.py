"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients_dict/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * build_experiment - build aggregator ready for federated learning simulation given arguments

"""

import json

from utils.args import *
from utils.utils import *

from torch.utils.tensorboard import SummaryWriter


def build_experiment(args_, seed_):
  with open(args_.cfg_file_path, "r") as f:
    all_clients_cfg = json.load(f)

  clients_dict = dict()
  n_samples_per_client = dict()

  print("\n==> Initialize Clients..")
  for client_id in tqdm(all_clients_cfg.keys(), position=0):
    data_dir = all_clients_cfg[client_id]["task_dir"]

    logs_dir = os.path.join(args_.logs_dir, f"client_{client_id}")
    os.makedirs(logs_dir, exist_ok=True)
    logger = SummaryWriter(logs_dir)

    clients_dict[int(client_id)] = init_client(
      args=args_, client_id=client_id, data_dir=data_dir, logger=logger
    )
    n_samples_per_client[client_id] = clients_dict[int(client_id)].num_samples

  clients_weights_dict = get_clients_weights(
    objective_type=args_.objective_type, n_samples_per_client=n_samples_per_client
  )

  global_trainer = get_trainer(
    experiment_name=args_.experiment,
    device=args_.device,
    optimizer_name=args_.server_optimizer,
    lr=args_.server_lr,
    seed=seed_,
  )

  global_logs_dir = os.path.join(args_.logs_dir, "global")
  os.makedirs(global_logs_dir, exist_ok=True)
  global_logger = SummaryWriter(global_logs_dir)

  aggregator_ = get_aggregator(
    aggregator_type=args_.aggregator_type,
    clients_dict=clients_dict,
    clients_weights_dict=clients_weights_dict,
    global_trainer=global_trainer,
    logger=global_logger,
    verbose=args_.verbose,
    seed=seed_,
  )

  return aggregator_


if __name__ == "__main__":
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  args = parse_args()

  seed = args.seed if (("seed" in args) and (args.seed >= 0)) else int(time.time())
  torch.manual_seed(seed)

  print("\n=> Build aggregator..")
  aggregator = build_experiment(args_=args, seed_=seed)

  aggregator.write_logs()

  print("\n=>Training..")

  for ii in tqdm(range(args.n_rounds)):

    aggregator.mix()

    if (ii % args.log_freq) == (args.log_freq - 1):
      aggregator.write_logs()
