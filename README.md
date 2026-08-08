# Neuromorphic Sandbox

A spiking neural network simulation framework for exploring spatial topology in AI computing.

## Quick Start

Navigate to the project and activate the virtual environment:

```powershell
cd C:\Users\Bo\projects\neuromorphic-sandbox
.\venv\Scripts\Activate.ps1
```

Then run Jupyter Lab:

```powershell
jupyter lab
```

Open your browser to `http://localhost:8888`

## Project Structure

- `src/` - Core simulation code
  - `simulation.py` - SNN simulator
  - `topology.py` - Spatial arrangement tools
  - `analysis.py` - Efficiency metrics
- `experiments/` - Jupyter notebooks for experiments
- `notebooks/` - Exploration notebooks
- `data/` - Simulation results and data

## Key Concepts

### Spatial Topology
This framework explores how physical arrangement of neurons affects computational efficiency, inspired by brain architecture.

### Spiking Neural Networks (SNNs)
Event-driven neural computation where neurons fire spikes based on membrane potential.

## Next Steps

1. Activate venv: `.\venv\Scripts\Activate.ps1`
2. Start Jupyter: `jupyter lab`
3. Create a new Python notebook
4. Import and experiment:

```python
from src.simulation import BasicNeuromorphicSim
from src.topology import TopologyGenerator

# Create and run a simulation
sim = BasicNeuromorphicSim(n_neurons=5000)
sim.create_network()
results = sim.run(duration=1.0)
print(f"Firing rate: {results['firing_rate']:.2f} Hz")
```

## Dependencies

- Brian2: Spiking neural network simulator
- NumPy/SciPy: Numerical computation
- PyTorch: Machine learning integration
- Matplotlib: Visualization
- Jupyter: Interactive notebooks

## Resources

- Brian2 docs: https://brian2.readthedocs.io/
- Neuromorphic computing: Nature Communications papers
