import numpy as np
from typing import Dict
import time

class BasicNeuromorphicSim:
    """Pure NumPy spiking neural network simulator (no Brian2 dependency)"""
    
    def __init__(self, n_neurons: int = 5000, dt: float = 0.1):
        self.n_neurons = n_neurons
        self.dt = dt / 1000.0  # Convert ms to seconds
        
        # Neuron state
        self.v = np.zeros(n_neurons)  # Membrane potential
        self.spike_history = []  # (time, neuron_id)
        self.v_history = []
        
    def create_network(self, connection_prob: float = 0.1):
        """Create a recurrent spiking neural network"""
        # Recurrent weights
        self.W_recurrent = (np.random.rand(self.n_neurons, self.n_neurons) < connection_prob).astype(float) * 0.01
        np.fill_diagonal(self.W_recurrent, 0)  # No self-connections
        
        # Input weights
        n_inputs = 100
        self.W_input = (np.random.rand(n_inputs, self.n_neurons) < 0.5).astype(float) * 0.1
        
        # Last spike time for each neuron (for refractory period)
        self.last_spike = np.ones(self.n_neurons) * -1000
        
    def run(self, duration: float = 1.0) -> Dict:
        """Run simulation and return metrics"""
        start_time = time.time()
        
        n_steps = int(duration / self.dt)
        tau = 10 * self.dt  # Time constant (10ms)
        
        spike_times = []
        spike_neurons = []
        
        for step in range(n_steps):
            # External input (Poisson-like)
            input_current = np.random.poisson(0.1, self.n_neurons) * 0.5
            
            # Current step
            t_current = step * self.dt
            
            # Decay voltage
            self.v = self.v * np.exp(-self.dt / tau)
            
            # Add input
            self.v += input_current
            
            # Add recurrent input from recent spikes
            if len(spike_neurons) > 0:
                # Get spikes from last timestep
                recent_spikes = np.array(spike_neurons[-self.n_neurons:])
                if len(recent_spikes) > 0:
                    input_from_spikes = self.W_recurrent[recent_spikes, :].sum(axis=0)
                    self.v += input_from_spikes
            
            # Check for spikes (threshold at 1.0)
            spiking = self.v > 1.0
            spike_indices = np.where(spiking)[0]
            
            # Reset spiking neurons
            self.v[spiking] = 0
            
            # Record spikes
            for neuron_id in spike_indices:
                spike_times.append(t_current)
                spike_neurons.append(neuron_id)
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        num_spikes = len(spike_times)
        firing_rate = num_spikes / (duration * self.n_neurons) if num_spikes > 0 else 0
        
        return {
            'duration': duration,
            'elapsed_time': elapsed,
            'num_spikes': num_spikes,
            'firing_rate': firing_rate,
            'spike_times': np.array(spike_times),
            'spike_neurons': np.array(spike_neurons)
        }


if __name__ == '__main__':
    print("Testing NumPy-based SNN simulator...")
    sim = BasicNeuromorphicSim(n_neurons=1000)
    sim.create_network()
    results = sim.run(duration=0.5)
    
    print(f"✓ Simulation complete!")
    print(f"  Firing rate: {results['firing_rate']:.4f} Hz")
    print(f"  Total spikes: {results['num_spikes']}")
    print(f"  Simulation time: {results['elapsed_time']:.2f}s")
