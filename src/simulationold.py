import brian2 as b2
import numpy as np
from typing import Dict
import time

class BasicNeuromorphicSim:
    """Baseline spiking neural network simulator"""
    
    def __init__(self, n_neurons: int = 5000, dt: float = 0.1):
        self.n_neurons = n_neurons
        self.dt = dt * b2.ms
        b2.start_scope()
        b2.defaultclock.dt = self.dt
        
    def create_network(self, connection_prob: float = 0.1):
        """Create a simple recurrent SNN"""
        # Neuron group with LIF (Leaky Integrate-and-Fire) model
        self.neurons = b2.NeuronGroup(
            self.n_neurons,
            '''dv/dt = (I - v) / (10*ms) : 1
               I : 1''',
            threshold='v>1',
            reset='v=0',
            method='exponential_euler'
        )
        
        # External input
        self.input_neuron = b2.PoissonGroup(100, rates=10*b2.Hz)
        self.input_syn = b2.Synapses(self.input_neuron, self.neurons, 'w:1')
        self.input_syn.connect(p=0.5)
        self.input_syn.w = 0.1
        
        # Recurrent connections
        self.recurrent = b2.Synapses(self.neurons, self.neurons, 'w:1')
        self.recurrent.connect(p=connection_prob)
        self.recurrent.w = 0.01
        
        # Monitor spikes
        self.spike_mon = b2.SpikeMonitor(self.neurons)
        self.state_mon = b2.StateMonitor(self.neurons, 'v', record=True)
        
    def run(self, duration: float = 1.0) -> Dict:
        """Run simulation and return metrics"""
        start_time = time.time()
        b2.run(duration * b2.second)
        elapsed = time.time() - start_time
        
        return {
            'duration': duration,
            'elapsed_time': elapsed,
            'num_spikes': len(self.spike_mon.spike_trains()),
            'firing_rate': len(self.spike_mon.t) / (duration * self.n_neurons),
            'spike_mon': self.spike_mon,
            'state_mon': self.state_mon
        }
