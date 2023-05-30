import itertools
import copy
import numpy as np
from scipy.optimize import LinearConstraint, minimize, Bounds, linprog
from remove_zero_rows import my_remove_redundancy_pivot_dense
from typing import List

class Graph:
    def __init__(self, network: dict, nbr_nodes: int):
        self.network : dict = network
        self.nbr_nodes: int = nbr_nodes
        self.nbr_edges: int = len(self.network.keys())
        self.paths: list = []

    def set_in_out_edges(self):
        self.in_edges = {}
        self.out_edges = {}
        for item in self.network.items():
            edge_id = int(item[0])
            node_out = item[1][0]
            node_in = item[1][1]
            if node_in in self.in_edges.keys():
                self.in_edges[node_in].append(edge_id)
            else:
                self.in_edges[node_in] = [edge_id]
            if node_out in self.out_edges.keys():
                self.out_edges[node_out].append(edge_id)
            else:
                self.out_edges[node_out] = [edge_id]

    def set_neighbor_nodes(self):
        self.neighbor_nodes = {}
        for node in range(self.nbr_nodes-1):
            self.neighbor_nodes[node] = []
            for edge_out in self.out_edges[node]:
                dest_node = self.network[str(edge_out)][1]
                if dest_node not in self.neighbor_nodes[node]:
                    self.neighbor_nodes[node].append(dest_node)

    def set_constraint(self, mu):
        # Out of Start Nodes Constraint
        row_start = np.zeros(self.nbr_edges)
        row_start[self.out_edges[0]] = 1
        limits = np.array([1])

        # In of Target Node Constraint
        row_end = np.zeros(self.nbr_edges)
        row_end[self.in_edges[self.nbr_nodes-1]] = 1
        limits = np.hstack([limits, np.array([1])])

        # Mass Conservation
        rows = []
        for node_id in self.in_edges.keys():
            if (node_id in self.out_edges.keys()):
                row = np.zeros(self.nbr_edges)
                row[self.in_edges[node_id]] = 1
                row[self.out_edges[node_id]] = -1
                if node_id != 0 and node_id != self.nbr_nodes-1:
                    rows.append(row)
                    limits = np.hstack([limits, np.array([0])])
        self.constraint_matrix = np.vstack([row_start, row_end, np.array(rows)]) #, np.eye(self.nbr_edges)])
        #self.lower_limit = np.hstack([limits, mu*np.ones(self.nbr_edges)])
        #self.upper_limit = np.hstack([limits, np.ones(self.nbr_edges)])


        A, keep_ind = my_remove_redundancy_pivot_dense(self.constraint_matrix)
        self.linear_constraint = LinearConstraint(A, limits[keep_ind], limits[keep_ind]) #self.lower_limit[keep_ind], self.upper_limit[keep_ind])
        self.A = A
        self.limits = limits[keep_ind]
        self.bound_constraint = Bounds(mu, 1)

    def set_bound_constraint(self, mu):
        self.bound_constraint = Bounds(mu, 1)

    def f(self, i, target):
        if i == target:
            return [[target]]
        return [[i] + p for c in self.neighbor_nodes[i] for p in self.f(c, target)]

    def set_all_paths(self, start, target):
        self.start = start
        self.target = target
        node_paths = self.f(start, target)
        for node_path in node_paths:
            edges_tot = []
            for j,node in enumerate(node_path[:-1]):
                edges = []
                for item in self.network.items():
                    edge_id = int(item[0])
                    node_out = item[1][0]
                    node_in = item[1][1]
                    if node_out == node and node_in == node_path[j+1]:
                        edges.append(edge_id)
                    
                edges_tot.append(edges)
            for element in itertools.product(*edges_tot):
                self.paths.append(list(element))
            
def compute_projection(y: np.ndarray, g: Graph, bregman: bool = False) -> np.ndarray:
    if bregman:
        dist = lambda x : np.sum(x*np.log(x/(y)))
        tol = 1e-3
    else:    
        dist = lambda x : np.linalg.norm(x - y)
        tol = None
    output = minimize(dist, x0=y, constraints = [g.linear_constraint], bounds= g.bound_constraint, method="SLSQP", tol=tol) #, method = "trust-constr", hess= lambda x: np.zeros((g.nbr_edges, g.nbr_edges)))
    if not output.success:
        raise Exception(f"Optimization Problem: {output.message}") 
    return output.x

def evaluate(paths, loads):
    output = [cost_function(loads[path]).sum() for path in paths]
    return output


def carat_sgd_multi_player(x0: np.ndarray, g: Graph, T: int, n: int, exp2: bool=False, gamma0: float = 0.1, mu0: float = 0.001, exp_graph:bool=False):
    cost_history = []
    paths_list = []
    x = np.zeros((n, g.nbr_edges))
    mu0 = 1/g.nbr_edges if not exp2 else 0
    if exp2: x0 += 1e-8
    g.set_bound_constraint(mu0)
    for agent in range(n):
        x[agent] = compute_projection(x0[agent], g, bregman=exp2)
    for t in range(T):
        if not exp2:
            mu_t = min([ 1/g.nbr_edges, mu0/(t+1)**(1/5)])
            g.set_bound_constraint(mu_t)
        gamma_t = gamma0/(t+1)**(3/5) if not exp2 else gamma0/(t+1)**(1/2)
        paths = []
        print(t)
        for agent in range(n):
            paths.append(sample_path(copy.deepcopy(x[agent]), g, exp_graph))
        loads = compute_edges_load(paths, g.nbr_edges)
        cost_history.append(evaluate(paths, loads))
        paths_list.append(paths)
        for agent in range(n):
            grad_estimate = estimate_cost(loads, paths[agent], x[agent])
            if exp2:
                y = x[agent]*np.exp(-gamma_t*grad_estimate)
            else:
                y = x[agent] - gamma_t*grad_estimate
            x[agent] = compute_projection(y, g, bregman=exp2)
        if np.min(x) <= 0:
            raise Exception(f"Negative x: {np.min(x)}")
        print(x)
    return x, cost_history, paths_list

def fpl_multi_player(x0: np.ndarray, g: Graph, T: int, n: int, gamma0: float = 0.1):
    cost_history = []
    paths_list = []
    L = np.zeros_like(x0)
    for t in range(T):
        gamma_t = gamma0/(t+1)**(1/2)
        paths = []
        print(t)
        for agent in range(n):
            paths.append(fpl_sample_path(g, L[agent], gamma_t))
        loads = compute_edges_load(paths, g.nbr_edges)
        paths_list.append(paths)
        cost_history.append(evaluate(paths, loads))
        for agent in range(n):
            grad_estimate = fpl_estimate_cost(loads, g ,L[agent], paths[agent], gamma=gamma_t, M=np.sqrt(t+1))
            L[agent] = L[agent] + grad_estimate
    return cost_history, paths_list

def fpl_sample_path(g: Graph, L: np.ndarray, gamma: float):

    Z = np.random.exponential(size=L.shape)
    x = linprog(gamma*L - Z, A_eq = g.A, b_eq= g.limits, bounds=(0,1))
    return np.where(x.x > 1e-6)[0]

def fpl_estimate_cost(loads: List[int], g: Graph, L: np.ndarray, path: List[int], gamma:float, M:float):

    K = np.zeros_like(L)
    estimate = np.zeros_like(L)
    estimate[path] = cost_function(loads[path])
    for k in range(int(M)):
        if len(path):
            p = fpl_sample_path(g,L, gamma)
            indices_to_update = [i for i in path if i in p]
            K[indices_to_update] = k+1
            path = [i for i in path if i not in indices_to_update]
        else:
            break
    if len(path):
        K[path] = M+1
    return estimate*K

def carat_sgd_adversarial_losses(x0: np.ndarray, g: Graph, T: int, distribution="Bernoulli", exp2: bool=True):
    cost_history = []
    mu0 = 1/g.nbr_edges if not exp2 else 0
    g.set_constraint(mu0)
    x = compute_projection(x0, g, bregman=exp2)
    for t in range(T):
        mu_t = min([ 1/g.nbr_edges, 0.001/(t+1)**(1/4)]) if not exp2 else 0
        g.set_constraint(mu_t)
        gamma_t = 0.1/(t+1)**(3/4) if not exp2 else 0.1/(t+1)**(1/2)
        print(t)
        path = sample_path(copy.deepcopy(x), g)
        print(path)
        if distribution == "Bernoulli":
            cost_t = bernoulli_adversarial_cost(t, g.nbr_edges)
        elif distribution == "Gaussian":
            cost_t = adversarial_cost(t, g.nbr_edges)
        cost_history.append(np.sum(cost_t[path]))
        grad_estimate = estimate_cost(cost_t, path, x)
        if exp2:
            y = x*np.exp(-gamma_t*grad_estimate)
        else:
            y = x - gamma_t*grad_estimate
        
        x = compute_projection(y, g, bregman=exp2)
        if np.min(x) <= 0:
            raise Exception(f"Negative x: {np.min(x)}")
    return x, cost_history

def exp3IX(x0: np.ndarray, g: Graph, T: int, distribution="Bernoulli"):
    cost_history = []
    K = len(g.paths)
    p = 1/K*np.ones(K)
    for t in range(T):
        eta_t = np.sqrt(np.log(K)/K/t)
        gamma_t = eta_t/2
        print(t)
        index_path = np.random.choice(np.arange(K),p=p)
        path = g.paths[index_path]
        print(path)
        if distribution == "Bernoulli":
            cost_t = bernoulli_adversarial_cost(t, g.nbr_edges)
        elif distribution == "Gaussian":
            cost_t = adversarial_cost(t, g.nbr_edges)
        cost_history.append(np.sum(cost_t[path]))
        grad_estimate = np.zeros(K)
        grad_estimate[index_path] = cost_t[path]/(p[index_path] + gamma_t)
        x = np.softmax(- eta_t*grad_estimate + np.log(x))
    return x, cost_history

def adversarial_cost(t, nbr_edges):

    costs = np.ones(nbr_edges)
    costs = costs + np.random.normal(0, 0.0001, size=(nbr_edges))
    if t > 5000:
        costs[0] = 0
        costs[1] = 0
    else:
        costs[11] = 0
        costs[14] = 0
    return costs

def bernoulli_adversarial_cost(t, nbr_edges):
    costs = 0.5*np.ones(nbr_edges)
    if t > 500000:
        costs[0] = 0.2
        costs[1] = 0.2
    else:
        costs[11] = 0.2
        costs[14] = 0.2
    costs = np.random.binomial(n=1,p=costs)
    return costs

def find_path_with_indices(g, among, must=None):
    for path in g.paths:
        if all(edges in among for edges in path):
            if must in path:
                return path, True
    return None, False

def find_path_with_indices_fast(g, among, must=None):
    node = g.start
    path = []
    while (node != g.target):
        next_edges = [a for a in among if a in g.out_edges[node]]
        if next_edges:
            if must in next_edges:
                next_edge = must
            else:
                next_edge = np.random.choice(next_edges)
            path.append(next_edge)
        else:
            return None, False
        node = g.network[str(next_edge)][1]
    return path, True


def efficient_decomposition(x: np.ndarray, g: Graph, fast: bool = True):
    basis = []
    probs = []
    found = True
    while found:
        valid_idx = np.where(x > 0)[0]
        if all(x == 0):
            break
        e_min = valid_idx[x[valid_idx].argmin()]
        x_min = x[e_min]
        if fast:
            path, found = find_path_with_indices_fast(g, among=valid_idx, must=e_min)
        else:
            path, found = find_path_with_indices(g, among=valid_idx, must=e_min)
        x[path] = x[path] - x_min
        if found:
            basis.append(path)
            probs.append(x_min)
    return basis, probs/np.sum(probs) #This division is just for normalization purposes

def sample_path(x: np.ndarray, g: Graph, fast: bool=False):
    basis, probs = efficient_decomposition(x,g, fast)
    id = np.random.choice(np.arange(len(basis)), p=probs)
    return basis[id]

def compute_edges_load(paths, nbr_edges):
    loads = []
    for e in range(nbr_edges):
        load = 0
        for path in paths:
            for edge in path:
                if edge == e:
                    load += 1
        loads.append(load)
    return np.array(loads)

def cost_function(load):
    return load

def estimate_cost(loads, path, x):
    estimate = np.zeros_like(x)
    estimate[path] = cost_function(loads[path])/x[path]
    return estimate











            

            
