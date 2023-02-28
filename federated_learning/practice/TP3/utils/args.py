import torch
import argparse
import warnings


def parse_args(args_list=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '--experiment',
        help='name of the experiment, possible are:'
             '{"mnist"}',
        type=str,
        required=True
    )
    parser.add_argument(
        '--method',
        help='the method to be used;'
             ' possible are `fedavg`, `perfedavg` ;',
        type=str,
        default="fedavg"
    )
    parser.add_argument(
        "--cfg_file_path",
        help='path to configuration file storing the configuration of each client',
        type=str,
        required=True
    )
    parser.add_argument(
        '--objective_type',
        help='type of the objective function; possible are:'
             '{"average", "weighted"}; default is "average"',
        type=str,
        default="average"
    )
    parser.add_argument(
        '--aggregator_type',
        help='aggregator type; possible are:'
             '{"centralized"}',
        type=str,
        default="centralized"
    )
    parser.add_argument(
        '--clients_sampler',
        help='type of clients_dict\' sampler; possible are:'
             '{"unbiased", "biased"};'
             'default is "unbiased"',
        type=str,
        default="unbiased"
    )
    parser.add_argument(
        '--n_rounds',
        help='number of communication rounds; default is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--local_steps',
        help='number of local steps before communication; default is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        '--local_optimizer',
        help='optimizer to be used for local training at clients_dict; default is sgd',
        type=str,
        default="sgd"
    )
    parser.add_argument(
        "--local_lr",
        type=float,
        help='learning rate for local training at clients_dict; default is 1e-3',
        default=1e-3
    )
    parser.add_argument(
        '--server_optimizer',
        help='server optimizer; default is sgd',
        type=str,
        default="sgd"
    )
    parser.add_argument(
        "--server_lr",
        type=float,
        help='server learning rate; default is 1.',
        default=1.
    )
    parser.add_argument(
        "--train_bz",
        type=int,
        help='batch size used for train, default is 1.',
        default=1
    )
    parser.add_argument(
        "--test_bz",
        type=int,
        help='batch size used for test and validation, default is 512.',
        default=512
    )
    parser.add_argument(
        '--device',
        help='device to use, either cpu or cuda; default is cpu',
        type=str,
        default="cpu"
    )
    parser.add_argument(
        '--log_freq',
        help='frequency of writing logs; defaults is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        "--verbose",
        help='verbosity level, `0` to quiet, `1` to show global logs and `2` to show local logs;'
             'default is `0`;',
        type=int,
        default=0
    )
    parser.add_argument(
        "--logs_dir",
        help='directory to write logs;',
        required=True
    )
    parser.add_argument(
        "--history_path",
        help="file to save the history of the clients sampler; expected to be a .json file."
             "if not passed, history is not saved",
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--seed",
        help='random seed; if not specified the system clock is used to generate the seed',
        type=int,
        default=argparse.SUPPRESS
    )

    if args_list:
        args = parser.parse_args(args_list)
    else:
        args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        warnings.warn("CUDA is not available, device is automatically set to \"CPU\"!", RuntimeWarning)

    return args
