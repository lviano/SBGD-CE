# Code for the ICML 2023 papers Semi-Bandit dynamics in Congestion Games

Run the script

```
python train_long.py 
```

to simulate the environment described in the paper. A chain with 20 states. The number of agents and the topology of the graph can be changed using the configuration file `long_graph.yaml`.

We already provide examples of different graph topologies in the config files `large_graph.yaml` and `graph.yaml`. To run SBGD-CE on these graphs use the scrips `train_large.py` and `train.py` respectively.

 