# Linear Regression
## Cancer Mortality Rate Analysis
### Project Overview
This project conducts a comprehensive linear regression analysis to examine the cancer mortality rate across various countries. We aim to understand the influence of different country-specific and population characteristics on cancer mortality rates using linear regression models.


### Objective
Our primary goal is to develop a predictive model that accurately forecasts cancer mortality rates based on a set of explanatory variables. This model will help in understanding the significant factors contributing to cancer mortality and provide insights for healthcare planning and policy-making.

### Data Preprocessing
Data preprocessing involved several steps to prepare the dataset for analysis:

+ __Variable Removal__: We evaluated the necessity of each variable in our model. Variables with insignificant impact on the explained variable (cancer mortality rate) were removed.
+ __Variable Transformation__: Continuous variables were examined for correlation, and categorical variables were analyzed through scatter plots to understand their relationship with the cancer mortality rate. Necessary transformations were applied to fit the regression model adequately.
+ __Dummy and Interaction Variables__: To incorporate categorical variables into the regression model, we created dummy variables and interaction terms to capture the combined effects of different variables on the cancer mortality rate.


### Model Selection and Evaluation
We utilized various algorithms such as Forward Selection, Backward Elimination, and Stepwise Regression to select the most relevant variables for our regression model based on AIC and BIC criteria. The model's assumptions were thoroughly tested to ensure its reliability and validity.

### Model Improvement
We explored different linear transformations of the dependent variable to enhance the model's performance. The natural logarithm transformation of the dependent variable yielded the best improvement in terms of the adjusted R-squared value.

### Conclusion
The final model provides valuable insights into the factors affecting cancer mortality rates and their interactions. This analysis can contribute to targeted healthcare interventions and policies aimed at reducing cancer mortality.

### Appendices
+ Variable definitions and descriptions
+ Pearson correlation coefficients for continuous variables
+ Results of algorithm runs for model selection
+ Statistical tests for model assumptions
