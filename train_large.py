import os
import hydra
import pickle
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import LinearConstraint
from graph import Graph, compute_projection, sample_path, carat_sgd_multi_player

def get_args(cfg: DictConfig):
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg

@hydra.main(config_path="config", config_name="large_graph")
def main(cfg: DictConfig):
    args = get_args(cfg)
    np.random.seed(args.seed)
    env = Graph(args.network.edges, args.network.nodes_nbr)
    env.set_in_out_edges()
    env.set_neighbor_nodes()
    env.set_constraint(mu=0)
    env.set_all_paths(start=0, target=args.network.nodes_nbr-1)
    x = compute_projection(1/env.nbr_edges*np.ones(env.nbr_edges), env)
    sample_path(x, env)
    if args.unif_init:
        x0 = np.array(args.n_agent*[1/env.nbr_edges*np.ones(env.nbr_edges)])
    else:
        x0 = np.hstack([np.array(args.n_agent*[[1,1]]), np.zeros((args.n_agent,23))])
    final_iterate, cost_history, _ = carat_sgd_multi_player(x0, env, 10000, args.n_agent, exp2=args.exp2, gamma0=3e-2, mu0=1e-6)
    if not args.exp2:
        if not args.unif_init:
            with open(hydra.utils.to_absolute_path(f"results/large_graph_{args.seed}.pkl"), "wb") as f:
                pickle.dump({"final_iterate":final_iterate,"cost_history":cost_history}, f)
        else:
            with open(hydra.utils.to_absolute_path(f"results/large_graph_{args.seed}_unif_init.pkl"), "wb") as f:
                pickle.dump({"final_iterate":final_iterate,"cost_history":cost_history}, f)

    else:
        if args.unif_init:
            with open(hydra.utils.to_absolute_path(f"results/large_graph_{args.seed}_exp2_unif_init.pkl"), "wb") as f:
                pickle.dump({"final_iterate":final_iterate,"cost_history":cost_history}, f)
        else:
            with open(hydra.utils.to_absolute_path(f"results/large_graph_{args.seed}_exp2.pkl"), "wb") as f:
                pickle.dump({"final_iterate":final_iterate,"cost_history":cost_history}, f)

if __name__ == "__main__":
    main()