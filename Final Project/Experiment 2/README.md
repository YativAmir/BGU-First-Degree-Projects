# Experiment 2: Network Depth Optimization

This experiment investigates the impact of network depth on performance for both traditional and ensemble approaches.

## Files
- `basic_network_testing.py`: Tests different layer depths for basic network
- `ensemble_network_testing.py`: Tests different layer depths for ensemble network

## Methodology
- Tests network depths from 10 to 40 layers (basic network)
- Tests network depths from 10 to 60 layers (ensemble network)
- 8 runs per configuration for statistical significance
- All other parameters kept constant:
  - 10 neurons per layer
  - BFGS optimizer
  - ReLU activation function

## Key Findings
### Basic Network
- Optimal depth: 10 hidden layers
- Average RMSE at optimal depth: 32.779
- Performance degrades with increasing depth

### Ensemble Network
- Optimal depth: 40 hidden layers
- Best performance at 40 layers
- Significant performance degradation below and above optimal depth

## How to Run
```python
# Test basic network
python basic_network_testing.py

# Test ensemble network
python ensemble_network_testing.py
```
