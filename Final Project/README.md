# Neural Networks Ensemble Learning Project

A comparative study of traditional neural networks versus ensemble learning approaches for house price prediction, implementing both architectures from scratch in Python.

## Overview

This project implements and compares two different approaches to neural network learning:
1. A traditional fully-connected neural network with BFGS optimization
2. An ensemble learning approach combining multiple neural networks

The implementation focuses on predicting house prices using the Kaggle House Prices dataset, with particular attention to performance in overparameterized scenarios.

## Authors

- Amir Yativ
- Doron Levi

Project completed at Ben Gurion University under the guidance of Dr. Yakir Berchencko.

## Project Structure

```
final project/
│
├── Data Prep/
│   └── train_data_set.py            # Data preprocessing and feature engineering
│
├── Experiment 1/                     # Baseline comparison experiments
│   ├── Basic ANN/
│   │   ├── final ANN BFGS.py        # Traditional neural network implementation
│   │   └── ANN with BFGS library.py # Library-based implementation for validation
│   │
│   └── Ensemble/
│       ├── final ANN Ensemble.py     # Ensemble network implementation
│       └── ANN with Ensemble library.py # Library-based implementation for validation
│
├── Experiment 2/                     # Network depth optimization
│   ├── basic_network_testing.py      # Testing different layer depths for basic network
│   └── ensemble_network_testing.py   # Testing different layer depths for ensemble
│
└── Experiment 3/                     # Iteration optimization
    ├── default_network_iterations.py # Optimal iterations for default network
    └── ensemble_iterations.py        # Optimal iterations for ensemble network
```

## Features

- Custom implementation of neural networks from scratch
- Ensemble learning approach with multiple network averaging
- Comprehensive feature engineering and data preprocessing
- Three detailed experiments comparing different aspects of the networks:
  1. Basic implementation validation and performance comparison
  2. Network depth optimization
  3. Training iteration optimization

## Dataset

The project uses the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) dataset from Kaggle.

## Requirements

```
numpy
pandas
scipy
scikit-learn
```

## Key Implementation Details

### Neural Network Architecture
- Fully connected network with configurable hidden layers
- BFGS optimizer
- ReLU activation function
- Configurable number of neurons per layer (default: 10)

### Ensemble Approach
- Multiple neural networks with identical architecture
- Weighted averaging using β vector
- No pre-training or partial training options
- Matrix-based prediction combination

## Experimental Results

### Experiment 1: Implementation Validation
- Compared custom implementations against library-based implementations
- Validated the functionality of both traditional and ensemble approaches
- Measured performance using MSE metrics

### Experiment 2: Network Depth
- Optimal depth for traditional network: 10 hidden layers
- Optimal depth for ensemble network: 40 hidden layers
- Performance measured across 8 iterations for reliability

### Experiment 3: Training Iterations
- Traditional network: Optimal at 200 iterations
- Ensemble network: Best performance at 35 iterations
- Partial training showed improved performance in specific ranges

## Usage

1. Data Preparation:
```python
python "Data Prep/train_data_set.py"
```

2. Run experiments (in order):
```python
# Experiment 1
python "Experiment 1/Basic ANN/final ANN BFGS.py"
python "Experiment 1/Ensemble/final ANN Ensemble.py"

# Experiment 2
python "Experiment 2/basic_network_testing.py"
python "Experiment 2/ensemble_network_testing.py"

# Experiment 3
python "Experiment 3/default_network_iterations.py"
python "Experiment 3/ensemble_iterations.py"
```

## Key Findings

- The traditional neural network outperformed the ensemble approach in most scenarios
- Network depth significantly impacts performance, with optimal depths varying between approaches
- Partial training can improve performance but requires careful optimization
- The ensemble approach showed limitations in handling complex relationships in the data

## Note

This project was developed as part of academic research. The implementation focuses on understanding neural network behavior in overparameterized scenarios rather than achieving state-of-the-art performance.
