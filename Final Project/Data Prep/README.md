# Data Preparation

This folder contains the data preprocessing and feature engineering code for the house prices dataset.

## File Structure
- `train_data_set.py`: Main preprocessing script

## Data Processing Steps

1. Feature categorization and transformation:
   - One-hot encoding for categorical features (MSZoning, Alley, etc.)
   - Percentile transformation for numerical features (LotFrontage, LotArea, etc.)
   - Scale adjustment for quality features (OverallQual, OverallCond)
   - Binary conversion for specific features (CentralAir)

2. Feature removal:
   - Removed unnecessary features: Utilities, BsmtFullBath, MiscFeature

## Dataset
Uses the Kaggle House Prices dataset: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

## Usage
Run `train_data_set.py` to preprocess the raw data before running any experiments.
