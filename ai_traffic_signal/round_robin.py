"""Weighted round-robin utilities and simple network visual helpers."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def weighted_round_robin(weights, num_requests: int = 100) -> Tuple[list, np.ndarray, np.ndarray]:
    num_nodes = len(weights)
    node_inflow = np.zeros(num_nodes, dtype=int)
    node_outflow = np.zeros(num_nodes, dtype=int)
    current_node = -1
    current_weight = 0
    total_weight = sum(weights)

    for _ in range(num_requests):
        while True:
            current_node = (current_node + 1) % num_nodes
            if current_node == 0:
                current_weight -= total_weight
            current_weight += weights[current_node]
            if current_weight >= 0:
                break
        node_inflow[current_node] += 1
        node_outflow[current_node] += 1

    # For compatibility with previous code which returned a networkx graph,
    # return None for the graph here and keep inflow/outflow arrays.
    return None, node_inflow, node_outflow
