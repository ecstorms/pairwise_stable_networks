"""
Code to accompany "On the Probability of Existence of Pairwise Stable Networks in Large Random Network Formation Settings"

This module loads the counterexample of Example 1, and performs a brute-force verification that no pairwise stable network exists with the specified linking costs.

"""

import os
import math
import pickle
import itertools as it
from functools import partial
import multiprocessing as mp

import numpy as np
import enlighten
from graph_tool.all import Graph, shortest_distance

from base import generate_binary_strings, seq_to_graph, plot_heatmap, draw_graph
from typing import List, Tuple, Callable, Dict, Optional, Any


def exhaustive_search(
    n: int,
    payoff_func: Callable[..., float],
    all_graphs: Optional[List[str]] = None,
    num_cores: int = 50,
    chunksize: int = 1,
    **kwargs: Any,
) -> Optional[Graph]:
    """
    Perform an exhaustive search over all possible graph strings to find a stable graph.

    Args:
        n: Number of vertices in each graph.
        payoff_func: Function to compute payoff for a given vertex and graph.
        all_graphs: Optional list of binary string representations of graphs. If None, generate all.
        num_cores: Number of parallel processes to use.
        chunksize: Number of tasks to dispatch per process at a time.
        **kwargs: Additional keyword arguments to pass to payoff_func.

    Returns:
        A stable graph (as a Graph object) if found; otherwise, None.
    """
    # Prepare the update function for each candidate graph
    updater = partial(Myopic_Update_from_string, payoff_func=payoff_func, **kwargs)

    if all_graphs is None:
        total_edges = math.comb(n, 2)
        all_graphs = list(generate_binary_strings(total_edges))

    # Set up a pool of workers
    pool = mp.Pool(num_cores)

    # Progress counter
    manager = enlighten.get_manager()
    ticks = manager.counter(
        total=len(all_graphs),
        desc="Searching",
        unit="graphs",
        color="cyan",
    )

    # Iterate through candidate graphs
    for graph_str, stable in pool.imap(updater, all_graphs, chunksize=chunksize):
        ticks.update(incr=1)
        if stable:
            pool.close()
            pool.join()
            return graph_str

    # Clean up resources
    pool.close()
    pool.join()
    print("No stable graph found.")
    return None


def distance_benefit(
    i: int,
    G: Graph,
    delta: float,
    payoff_dict: Dict[Tuple[int, int], float],
    alpha: float = 1.0,
) -> float:
    """
    Calculate the distance-based benefit (utility) for a vertex in the graph.

    Utility is sum of discounted reachability benefits minus linking costs.

    Args:
        i: Index of the vertex.
        G: A graph_tool Graph object.
        delta: Discount factor for distance benefits.
        payoff_dict: Mapping of edge pairs to linking costs.
        alpha: Weight on linking cost.

    Returns:
        The computed utility (float).
    """
    utility: float = 0.0
    # For every other vertex, accumulate benefits and subtract costs
    for v in G.vertices():
        j = G.vertex_index[v]
        if j == i:
            continue
        # Add distance-based benefit
        dist = shortest_distance(G, i, j)
        utility += delta**dist
        # Subtract linking cost if edge exists
        if G.edge(i, j) is not None:
            cost = payoff_dict.get((i, j), 0.0)
            utility -= alpha * cost
    return utility


def Myopic_Update(
    G: Graph, payoff_func: Callable[..., float], **kwargs: Any
) -> Tuple[Graph, bool]:
    """
    Perform a single myopic update on the graph by adding/removing an edge.

    Randomly considers pairs of vertices for potential blocking links until stability.

    Args:
        G: A graph_tool Graph object to update.
        payoff_func: Function to compute payoffs for vertices.
        **kwargs: Additional parameters for payoff_func.

    Returns:
        A tuple of (updated Graph, stability flag).
            stability flag is False if an update occurred; True otherwise.
    """
    vertex_pairs = list(it.combinations(range(G.num_vertices()), 2))
    np.random.shuffle(vertex_pairs)
    stable = True

    for i, j in vertex_pairs:
        G_alt = G.copy()
        if G.edge(i, j) is None:
            G_alt.add_edge(i, j)
            better_i = payoff_func(i, G_alt, **kwargs) >= payoff_func(i, G, **kwargs)
            better_j = payoff_func(j, G_alt, **kwargs) >= payoff_func(j, G, **kwargs)
            if better_i and better_j:
                return G_alt, False
        else:
            G_alt.remove_edge(G.edge(i, j))
            improved_i = payoff_func(i, G_alt, **kwargs) > payoff_func(i, G, **kwargs)
            improved_j = payoff_func(j, G_alt, **kwargs) > payoff_func(j, G, **kwargs)
            if improved_i or improved_j:
                return G_alt, False

    return G, stable


def Myopic_Update_from_string(
    graph_str: str, payoff_func: Callable[..., float], **kwargs: Any
) -> Tuple[Graph, bool]:
    """
    Convert a binary string to a graph, then perform a myopic update.

    Args:
        graph_str: Binary string encoding of the graph.
        payoff_func: Function to compute payoffs for vertices.
        **kwargs: Additional parameters for payoff_func.

    Returns:
        A tuple of (updated Graph, stability flag).
    """
    G = seq_to_graph(graph_str)
    return Myopic_Update(G, payoff_func, **kwargs)


if __name__ == "__main__":
    # Set working directory and load example data
    os.chdir("../precomupted_examples/")

    core_count = mp.cpu_count()

    with open("nonexistence_8_example.pkl", "rb") as f:
        data = pickle.load(f)

    # Extract parameters from data
    delta: float = data[0]
    payoff_dict: Dict[Tuple[int, int], float] = data[-1][-1]

    # Display pairwise payoffs
    for (i, j), cost in payoff_dict.items():
        print(f"({i}, {j}): {cost}")

    # Load full payoff dictionary
    with open("nonexistence_8_example_full.pkl", "rb") as f:
        payoff_dict = pickle.load(f)

    # Construct payoff matrix
    A: np.ndarray = np.zeros((8, 8))
    for (i, j), cost in payoff_dict.items():
        A[i, j] = cost
        A[j, i] = payoff_dict.get((j, i), 0.0)

    # Plot heatmap of linking costs
    plot_heatmap(A, title="Counterexample Linking Costs", blackout_diagonal=True)

    # Generate all candidate graph strings for exhaustive search
    total_edges = math.comb(8, 2)
    graphs = list(generate_binary_strings(total_edges - 3))
    prefixes = ["111", "110", "101", "100", "011", "010", "001", "000"]
    all_graphs_list = [prefix + g for prefix in prefixes for g in graphs]

    # Run exhaustive searches (examples)
    search_result = exhaustive_search(
        n=8,
        payoff_func=distance_benefit,
        all_graphs=all_graphs_list,
        payoff_dict=payoff_dict,
        delta=delta,
        alpha=1.0,
        chunksize=20000,
        num_cores=core_count,
    )

    # Build and draw the graph for visualization
    edge_list = [[i, j, cost] for (i, j), cost in payoff_dict.items()]
    G = Graph()
    G.ep["linking_cost"] = G.new_edge_property("float")
    G.add_edge_list(edge_list, eprops=[G.ep["linking_cost"]])
    draw_graph(
        G,
        vertex_size=30,
        vertex_text=G.vertex_index,
        edge_color=G.ep["linking_cost"],
        make_edge_color_heatmap=True,
        edge_marker_size=10,
        heatmap_legend_color="#000000",
        make_edge_legend=True,
        output="nonexistence_8_example_full.png",
    )
