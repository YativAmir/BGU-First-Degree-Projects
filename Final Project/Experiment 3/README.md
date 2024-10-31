# Experiment 3: Iteration Optimization

This experiment examines the effect of partial training and iteration count on network performance.

## Files
- `default_network_iterations.py`: Tests optimal iterations for default network
- `ensemble_iterations.py`: Tests optimal iterations for ensemble network

## Methodology
### Default Network
- Tests iteration ranges: 50 to 500 (increments of 50)
- 10 runs per configuration
- Fixed architecture: 4 hidden layers, 10 neurons per layer

### Ensemble Network
- Tests iteration ranges: 5 to 65 (increments of 10)
- 5 runs per configuration
- Same architecture as default network

## Key Findings
### Default Network
- Optimal iterations: 200
- Lowest RMSE: 27.683
- Performance stabilizes after optimal point

### Ensemble Network
- Optimal iterations: 35
- Best RMSE: 1,571,627
- Performance degrades after 45 iterations

## How to Run
```python
# Test default network iterations
python default_network_iterations.py

# Test ensemble iterations
python ensemble_iterations.py
```
