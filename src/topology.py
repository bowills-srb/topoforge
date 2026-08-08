import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class SpatialNeuron:
    """Neuron with explicit 3D coordinates"""
    id: int
    x: float
    y: float
    z: float = 0.0
    
    def distance_to(self, other: 'SpatialNeuron') -> float:
        return np.sqrt((self.x - other.x)**2 + 
                      (self.y - other.y)**2 + 
                      (self.z - other.z)**2)

class TopologyGenerator:
    """Generate spatial arrangements for neurons"""
    
    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons
        self.neurons = []
        
    def grid_2d(self, width: int, height: int) -> List[SpatialNeuron]:
        """Arrange neurons in 2D grid"""
        self.neurons = []
        for i in range(self.n_neurons):
            x = (i % width) * 1.0
            y = (i // width) * 1.0
            self.neurons.append(SpatialNeuron(id=i, x=x, y=y))
        return self.neurons
    
    def columnar_3d(self, cols: int, rows: int, layers: int) -> List[SpatialNeuron]:
        """Brain-like columnar structure with layers"""
        self.neurons = []
        idx = 0
        for z in range(layers):
            for y in range(rows):
                for x in range(cols):
                    if idx < self.n_neurons:
                        self.neurons.append(SpatialNeuron(
                            id=idx, x=x, y=y, z=z
                        ))
                        idx += 1
        return self.neurons
    
    def random_cluster(self, n_clusters: int, cluster_radius: float = 2.0) -> List[SpatialNeuron]:
        """Random clusters (like cortical columns)"""
        self.neurons = []
        neurons_per_cluster = self.n_neurons // n_clusters
        
        for cluster_id in range(n_clusters):
            center_x = np.random.uniform(0, 10)
            center_y = np.random.uniform(0, 10)
            
            for i in range(neurons_per_cluster):
                if len(self.neurons) < self.n_neurons:
                    theta = np.random.uniform(0, 2*np.pi)
                    r = np.random.uniform(0, cluster_radius)
                    x = center_x + r * np.cos(theta)
                    y = center_y + r * np.sin(theta)
                    self.neurons.append(SpatialNeuron(
                        id=len(self.neurons), x=x, y=y
                    ))
        return self.neurons
    
    def proximity_connection_matrix(self, distance_threshold: float) -> np.ndarray:
        """Create connectivity based on spatial proximity"""
        adjacency = np.zeros((self.n_neurons, self.n_neurons))
        
        for i, neuron_i in enumerate(self.neurons):
            for j, neuron_j in enumerate(self.neurons):
                if i != j:
                    dist = neuron_i.distance_to(neuron_j)
                    if dist < distance_threshold:
                        adjacency[i, j] = 1.0
        
        return adjacency
    
    def get_coordinates(self) -> np.ndarray:
        """Return all neuron positions"""
        return np.array([[n.x, n.y, n.z] for n in self.neurons])
