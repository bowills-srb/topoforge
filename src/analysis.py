import numpy as np
from typing import Dict, Tuple

class SpatialCommunicationAnalyzer:
    """Analyze efficiency of spatial arrangements"""
    
    def __init__(self, neuron_coordinates: np.ndarray, adjacency_matrix: np.ndarray):
        self.coords = neuron_coordinates
        self.adjacency = adjacency_matrix
        self.n_neurons = len(neuron_coordinates)
        
    def average_connection_distance(self) -> float:
        """How far do signals travel on average?"""
        total_distance = 0
        num_connections = 0
        
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if self.adjacency[i, j] > 0:
                    dist = np.linalg.norm(self.coords[i] - self.coords[j])
                    total_distance += dist
                    num_connections += 1
        
        return total_distance / num_connections if num_connections > 0 else 0
    
    def local_vs_global_ratio(self, local_threshold: float = 2.0) -> Tuple[float, float]:
        """What fraction of connections are local vs. long-range?"""
        local_connections = 0
        global_connections = 0
        
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if self.adjacency[i, j] > 0:
                    dist = np.linalg.norm(self.coords[i] - self.coords[j])
                    if dist < local_threshold:
                        local_connections += 1
                    else:
                        global_connections += 1
        
        total = local_connections + global_connections
        return (local_connections / total if total > 0 else 0,
                global_connections / total if total > 0 else 0)
    
    def report(self) -> Dict:
        """Generate efficiency report"""
        return {
            'avg_connection_distance': self.average_connection_distance(),
            'local_ratio': self.local_vs_global_ratio()[0],
            'global_ratio': self.local_vs_global_ratio()[1],
        }
