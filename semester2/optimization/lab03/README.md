<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Part 1: Parameter Server](#part-1-parameter-server)
- [Part 2: Federated Averaging](#part-2-federated-averaging)
- [Multi-Node multi-process distributed training (e.g. two nodes)](#multi-node-multi-process-distributed-training-eg-two-nodes)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->



## Part 1: Parameter Server
To launch the program:

1. Go to data directory, run python `generate_data.py --n_clients 10`
2. Go back to the root directory, run
   
   ```python -m torch.distributed.launch --nproc_per_node=11 main.py --n_rounds 100```

## Part 2: Federated Averaging
To launch the program:

1. Go to data directory, run python `generate_data.py --n_iid --n_clients 10`
2. Go back to the root directory, run 
   
    ```python -m torch.distributed.launch --nproc_per_node=11  main.py --run_fedavg --n_rounds 100 --n_local_epochs 1```

## Multi-Node multi-process distributed training (e.g. two nodes)
Node 1: (IP: 192.168.1.1, and has a free port: 1234)
````
python -m torch.distributed.launch --nproc_per_node=5
        --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
        --master_port=1234 main.py --n_rounds 100
````

Node 2: 
````
python -m torch.distributed.launch --nproc_per_node=5
        --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
        --master_port=1234 main.py --n_rounds 100
````
