"""RRT* and Dijkstra utilities for spatial path planning.

This module is a cleaned, import-safe version of the original script.
Optional heavy dependencies (matplotlib, networkx) are only imported inside
functions that need them so lightweight import is possible.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np

class Node:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self.parent: Optional["Node"] = None
        self.cost: float = 0.0


def calculate_distance(node1: Node, node2: Node) -> float:
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2 + (node1.z - node2.z) ** 2)


def nearest_neighbor(random_node: Node, nodes: List[Node]) -> Node:
    min_dist = float("inf")
    nearest = nodes[0]
    for node in nodes:
        dist = calculate_distance(random_node, node)
        if dist < min_dist:
            min_dist = dist
            nearest = node
    return nearest


def extend_tree(from_node: Node, to_node: Node, step_size: float) -> Node:
    dist = calculate_distance(from_node, to_node)
    if dist <= step_size:
        return to_node
    direction_x = (to_node.x - from_node.x) / dist
    direction_y = (to_node.y - from_node.y) / dist
    direction_z = (to_node.z - from_node.z) / dist
    return Node(from_node.x + direction_x * step_size,
                from_node.y + direction_y * step_size,
                from_node.z + direction_z * step_size)


def rrt_star_algorithm(x_array: np.ndarray, y_array: np.ndarray, z_array: np.ndarray,
                       start_point: Tuple[float, float, float], end_point: Tuple[float, float, float],
                       num_iterations: int = 1000, step_size: float = 1.0, rewire_radius: float = 2.0):
    start_node = Node(*start_point)
    end_node_rep = Node(*end_point)
    nodes: List[Node] = [start_node]
    best_path = None
    min_path_cost = float("inf")

    min_x, max_x = float(np.min(x_array)), float(np.max(x_array))
    min_y, max_y = float(np.min(y_array)), float(np.max(y_array))
    min_z, max_z = float(np.min(z_array)), float(np.max(z_array))

    for _ in range(num_iterations):
        rx = random.uniform(min_x, max_x)
        ry = random.uniform(min_y, max_y)
        rz = random.uniform(min_z, max_z)
        random_node = Node(rx, ry, rz)

        nearest = nearest_neighbor(random_node, nodes)
        new_node = extend_tree(nearest, random_node, step_size)

        # Basic placeholder validity check
        if True:
            # Find best parent (simplified)
            new_node.parent = nearest
            new_node.cost = nearest.cost + calculate_distance(new_node, nearest)
            nodes.append(new_node)

            # Check if goal reached
            if calculate_distance(new_node, end_node_rep) <= step_size:
                path_cost = new_node.cost + calculate_distance(new_node, end_node_rep)
                if path_cost < min_path_cost:
                    min_path_cost = path_cost
                    # reconstruct
                    cur = new_node
                    path = [end_node_rep]
                    while cur is not None:
                        path.append(cur)
                        cur = cur.parent
                    path.reverse()
                    best_path = path

    return nodes, best_path, min_path_cost


def dijkstra_algorithm(x_array: np.ndarray, y_array: np.ndarray, z_array: np.ndarray,
                       start_node: Tuple[int, int], end_node: Tuple[int, int]):
    import heapq
    import networkx as nx

    rows, cols = x_array.shape
    G = nx.DiGraph()
    for r in range(rows):
        for c in range(cols):
            G.add_node((r, c))

    movements = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(rows):
        for c in range(cols):
            for dr, dc in movements:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    distance = math.sqrt((x_array[r, c] - x_array[nr, nc])**2 +
                                         (y_array[r, c] - y_array[nr, nc])**2 +
                                         (z_array[r, c] - z_array[nr, nc])**2)
                    G.add_edge((r, c), (nr, nc), weight=distance)

    distances = {node: float('inf') for node in G.nodes}
    distances[start_node] = 0
    pq = [(0, start_node)]
    predecessors = {}

    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_distance > distances[current_node]:
            continue
        if current_node == end_node:
            break
        for neighbor, attributes in G[current_node].items():
            weight = attributes['weight']
            d = current_distance + weight
            if d < distances[neighbor]:
                distances[neighbor] = d
                predecessors[neighbor] = current_node
                heapq.heappush(pq, (d, neighbor))

    path = []
    cur = end_node
    while cur in predecessors:
        path.append(cur)
        cur = predecessors[cur]
    if path:
        path.append(start_node)
        path.reverse()
        return path, distances[end_node]
    return None, None
