#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RRT* and Dijkstra's Algorithm Implementation without pandas
Using numpy arrays for data handling
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import heapq
import math
import random

class Node:
    """Represents a node in the RRT* tree."""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.cost = 0.0

def dijkstra_algorithm(x_array, y_array, z_array, start_node, end_node):
    """
    Implements Dijkstra's algorithm to find the shortest path in a 3D grid graph.
    
    Args:
        x_array (np.ndarray): Array containing x-coordinates.
        y_array (np.ndarray): Array containing y-coordinates.
        z_array (np.ndarray): Array containing z-coordinates.
        start_node (tuple): Tuple representing the start node (row, col).
        end_node (tuple): Tuple representing the end node (row, col).
    
    Returns:
        tuple: A tuple containing the shortest path and the total distance.
    """
    rows, cols = x_array.shape
    G = nx.DiGraph()
    
    # Add nodes to the graph
    for r in range(rows):
        for c in range(cols):
            G.add_node((r, c))
    
    # Add edges with weights based on distance between adjacent nodes
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for r in range(rows):
        for c in range(cols):
            current_node = (r, c)
            for dr, dc in movements:
                neighbor_r, neighbor_c = r + dr, c + dc
                if 0 <= neighbor_r < rows and 0 <= neighbor_c < cols:
                    neighbor_node = (neighbor_r, neighbor_c)
                    # Calculate Euclidean distance in 3D space
                    distance = np.sqrt(
                        (x_array[r, c] - x_array[neighbor_r, neighbor_c])**2 +
                        (y_array[r, c] - y_array[neighbor_r, neighbor_c])**2 +
                        (z_array[r, c] - z_array[neighbor_r, neighbor_c])**2
                    )
                    G.add_edge(current_node, neighbor_node, weight=distance)
    
    # Implement Dijkstra's algorithm
    distances = {node: float('inf') for node in G.nodes}
    distances[start_node] = 0
    priority_queue = [(0, start_node)]
    predecessors = {}
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
            
        if current_node == end_node:
            break
            
        for neighbor, attributes in G[current_node].items():
            weight = attributes['weight']
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    
    # Reconstruct the shortest path
    path = []
    current = end_node
    while current in predecessors:
        path.append(current)
        current = predecessors[current]
    if path:
        path.append(start_node)
        path.reverse()
        return path, distances[end_node]
    else:
        return None, None

def calculate_distance(node1, node2):
    """Calculates the Euclidean distance between two nodes in 3D."""
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2 + (node1.z - node2.z)**2)

def is_valid(node, x_array, y_array, z_array):
    """Checks if a node is valid (placeholder implementation)."""
    return True

def nearest_neighbor(random_node, nodes):
    """Finds the nearest node in the tree to the random node."""
    min_dist = float('inf')
    nearest = None
    for node in nodes:
        dist = calculate_distance(random_node, node)
        if dist < min_dist:
            min_dist = dist
            nearest = node
    return nearest

def extend_tree(from_node, to_node, step_size):
    """Extends from a node towards another node with a given step size."""
    dist = calculate_distance(from_node, to_node)
    if dist < step_size:
        return to_node
    else:
        direction_x = (to_node.x - from_node.x) / dist
        direction_y = (to_node.y - from_node.y) / dist
        direction_z = (to_node.z - from_node.z) / dist
        new_x = from_node.x + direction_x * step_size
        new_y = from_node.y + direction_y * step_size
        new_z = from_node.z + direction_z * step_size
        return Node(new_x, new_y, new_z)

def find_best_parent(new_node, nearest_node, nodes, rewire_radius, x_array, y_array, z_array):
    """Finds the best parent for the new node within the rewire radius."""
    min_cost = nearest_node.cost + calculate_distance(new_node, nearest_node)
    best_parent = nearest_node
    
    for node in nodes:
        if calculate_distance(new_node, node) < rewire_radius:
            if True:  # Placeholder for collision checking
                cost = node.cost + calculate_distance(new_node, node)
                if cost < min_cost:
                    min_cost = cost
                    best_parent = node
    new_node.cost = min_cost
    return best_parent

def rewire_neighbors(new_node, nodes, rewire_radius, x_array, y_array, z_array):
    """Rewires neighbors if the new node provides a shorter path."""
    for node in nodes:
        if node != new_node.parent and calculate_distance(new_node, node) < rewire_radius:
            if True:  # Placeholder for collision checking
                if new_node.cost + calculate_distance(new_node, node) < node.cost:
                    node.parent = new_node
                    node.cost = new_node.cost + calculate_distance(new_node, node)

def rrt_star_algorithm(x_array, y_array, z_array, start_point, end_point, num_iterations, step_size, rewire_radius):
    """
    Implements the RRT* algorithm.
    
    Args:
        x_array (np.ndarray): Array containing x-coordinates.
        y_array (np.ndarray): Array containing y-coordinates.
        z_array (np.ndarray): Array containing z-coordinates.
        start_point (tuple): Tuple representing the start point (x, y, z).
        end_point (tuple): Tuple representing the end point (x, y, z).
        num_iterations (int): The number of iterations to run the algorithm.
        step_size (float): The step size for extending the tree.
        rewire_radius (float): The radius for rewiring neighbors.
    
    Returns:
        tuple: A tuple containing the list of all nodes, the final path, and the path cost.
    """
    start_node = Node(start_point[0], start_point[1], start_point[2])
    end_node_representation = Node(end_point[0], end_point[1], end_point[2])
    nodes = [start_node]
    best_path = None
    min_path_cost = float('inf')
    
    # Determine the bounds of the state space
    min_x, max_x = np.min(x_array), np.max(x_array)
    min_y, max_y = np.min(y_array), np.max(y_array)
    min_z, max_z = np.min(z_array), np.max(z_array)
    
    for _ in range(num_iterations):
        # Randomly sample a point in the state space
        random_x = random.uniform(min_x, max_x)
        random_y = random.uniform(min_y, max_y)
        random_z = random.uniform(min_z, max_z)
        random_node = Node(random_x, random_y, random_z)
        
        # Find the nearest node in the current RRT* tree
        nearest = nearest_neighbor(random_node, nodes)
        
        # Extend from the nearest node towards the sampled point
        new_node = extend_tree(nearest, random_node, step_size)
        
        # Check if the new node is valid
        if is_valid(new_node, x_array, y_array, z_array):
            # Find the best parent for the new node
            best_parent = find_best_parent(new_node, nearest, nodes, rewire_radius, x_array, y_array, z_array)
            new_node.parent = best_parent
            nodes.append(new_node)
            
            # Rewire neighbors
            rewire_neighbors(new_node, nodes, rewire_radius, x_array, y_array, z_array)
            
            # Check if the new node is close to the end point
            if calculate_distance(new_node, end_node_representation) < step_size:
                if True:  # Placeholder for collision checking
                    path_cost = new_node.cost + calculate_distance(new_node, end_node_representation)
                    if path_cost < min_path_cost:
                        min_path_cost = path_cost
                        # Reconstruct and store the path
                        current_path_node = new_node
                        current_path = [end_node_representation]
                        while current_path_node is not None:
                            current_path.append(current_path_node)
                            current_path_node = current_path_node.parent
                        current_path.reverse()
                        best_path = current_path
    
    return nodes, best_path, min_path_cost

def plot_dijkstra_path(x_array, y_array, z_array, shortest_path):
    """Plots the shortest path found by Dijkstra's algorithm in 3D."""
    if not shortest_path:
        print("No Dijkstra path to plot.")
        return
    
    path_x = [x_array[node[0], node[1]] for node in shortest_path]
    path_y = [y_array[node[0], node[1]] for node in shortest_path]
    path_z = [z_array[node[0], node[1]] for node in shortest_path]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the grid points for context
    ax.scatter(x_array.flatten(), y_array.flatten(), z_array.flatten(), c='gray', alpha=0.5, label='Grid Points')
    
    # Plot the shortest path
    ax.plot(path_x, path_y, path_z, marker='o', c='red', label='Shortest Path')
    ax.scatter(path_x[0], path_y[0], path_z[0], c='green', marker='o', s=100, label='Start')
    ax.scatter(path_x[-1], path_y[-1], path_z[-1], c='blue', marker='o', s=100, label='End')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('Dijkstra\'s Shortest Path')
    ax.legend()
    plt.show()

def plot_rrt_tree(x_array, y_array, z_array, rrt_nodes, rrt_path):
    """Plots the RRT* tree and the found path in 3D."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the grid points for context
    ax.scatter(x_array.flatten(), y_array.flatten(), z_array.flatten(), c='gray', alpha=0.5, label='Grid Points')
    
    # Plot the RRT* tree edges
    for node in rrt_nodes:
        if node.parent:
            ax.plot([node.x, node.parent.x], [node.y, node.parent.y], [node.z, node.parent.z], c='cyan', alpha=0.5)
    
    # Plot the RRT* path if found
    if rrt_path:
        path_x = [node.x for node in rrt_path]
        path_y = [node.y for node in rrt_path]
        path_z = [node.z for node in rrt_path]
        ax.plot(path_x, path_y, path_z, marker='o', c='red', label='RRT* Path')
        ax.scatter(path_x[0], path_y[0], path_z[0], c='green', marker='o', s=100, label='Start')
        ax.scatter(path_x[-1], path_y[-1], path_z[-1], c='blue', marker='o', s=100, label='End')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('RRT* Tree and Path')
    ax.legend()
    plt.show()

def main():
    """Main function to run the algorithms and plot results."""
    # Create sample arrays for demonstration
    np.random.seed(42)  # For reproducible results
    data = np.random.rand(10, 10) * 100
    x_array = data.copy()
    y_array = data.copy() + np.random.rand(10, 10) * 20
    z_array = data.copy() + np.random.rand(10, 10) * 30
    
    print("Sample x_array (first 5x5):")
    print(x_array[:5, :5])
    print("\nSample y_array (first 5x5):")
    print(y_array[:5, :5])
    print("\nSample z_array (first 5x5):")
    print(z_array[:5, :5])
    
    # Define start and end points for Dijkstra's (grid indices)
    start_dijkstra = (0, 0)
    end_dijkstra = (9, 9)
    
    # Run Dijkstra's algorithm
    print("\nRunning Dijkstra's algorithm...")
    shortest_path, total_distance_dijkstra = dijkstra_algorithm(x_array, y_array, z_array, start_dijkstra, end_dijkstra)
    
    if shortest_path:
        print("Dijkstra's Shortest Path found with total distance: {:.2f}".format(total_distance_dijkstra))
        print("Path length: {} nodes".format(len(shortest_path)))
    else:
        print("No Dijkstra's path found.")
    
    # Define start and end points for RRT* (3D coordinates)
    start_point_rrt = (x_array[0, 0], y_array[0, 0], z_array[0, 0])
    end_point_rrt = (x_array[9, 9], y_array[9, 9], z_array[9, 9])
    
    # Set parameters for RRT*
    num_iterations = 1000
    step_size = 5.0
    rewire_radius = 10.0
    
    # Run RRT* algorithm
    print("\nRunning RRT* algorithm...")
    rrt_nodes, rrt_path, total_distance_rrt = rrt_star_algorithm(
        x_array, y_array, z_array, start_point_rrt, end_point_rrt, 
        num_iterations, step_size, rewire_radius
    )
    
    if rrt_path:
        print("RRT* Path found with cost: {:.2f}".format(total_distance_rrt))
        print("Path length: {} nodes".format(len(rrt_path)))
        print("Total nodes explored: {}".format(len(rrt_nodes)))
    else:
        print("No RRT* path found to the goal within the given iterations.")
        print("Total nodes explored: {}".format(len(rrt_nodes)))
    
    # Plot the results
    print("\nGenerating plots...")
    plot_dijkstra_path(x_array, y_array, z_array, shortest_path)
    plot_rrt_tree(x_array, y_array, z_array, rrt_nodes, rrt_path)
    
    print("Algorithm execution completed!")

if __name__ == "__main__":
    main()