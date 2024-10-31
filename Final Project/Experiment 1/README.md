
# Experiment 1: Implementation Validation

This experiment validates our custom neural network implementations against library-based versions to ensure correct functionality.

## Folder Structure
### Basic ANN
- `final ANN BFGS.py`: Custom implementation of traditional neural network
- `ANN with BFGS library.py`: Library-based implementation for validation

### Ensemble
- `final ANN Ensemble.py`: Custom implementation of ensemble network
- `ANN with Ensemble library.py`: Library-based implementation for validation

## Implementation Details
- Network Architecture: 4 hidden layers, 10 neurons per layer
- Optimizer: BFGS
- Activation Function: ReLU
- Performance Metric: Mean Squared Error (MSE)

## Results
Results show comparison between:
1. Custom BFGS vs Library BFGS
   - Custom: Average MSE of 45,376
   - Library: Average MSE of 29,7979

2. Custom Ensemble vs Library Ensemble
   - Custom: Average MSE of 6,837,620
   - Library: Average MSE of 3,755,090

## How to Run
1. Run basic ANN comparison:
```python
python "Basic ANN/final ANN BFGS.py"
python "Basic ANN/ANN with BFGS library.py"
```

2. Run ensemble comparison:
```python
python "Ensemble/final ANN Ensemble.py"
python "Ensemble/ANN with Ensemble library.py"
```
