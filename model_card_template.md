# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A logistic regression model was used as a binary classification model for 
salary in the Census data
- Model type: sklearn.linear_model.LogisticRegression
- Model date: 13.02.2022
- Parameters: max_iter=300
- Categorical features: OneHot encoding was used
- Continuous features: Standard scaler was used

## Intended Use
This model is intended to be used as part of the Udacity project to classify
salary in the Census data provided

## Training Data
The Census data provided by UCI was used:
https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
The Census data was split into training and test with test_size = 20%

## Metrics
Metrics used are:
- Precision: 0.72
- Recall: 0.60
- F-Score: 0.66

For slice performances two additional metrics were added:
- TNR: True Negative Rate
- NPR: Negative Predictive Value

The slice performances are included in slice_output.txt

## Ethical Considerations
No ethical consideration to be considered here to the best of my knowledge.

## Caveats and Recommendations
Alternative models could be tested to further improve the performance.
Additionally, hyper-parameter tuning was not considered here as the model which
was used is a logistic regression model. However, hyper-parameter tuning should
be considered when other models are used, such as random forests. Then, the data
should be split into training, evaluation, and test. If the used model is not 
computationally expensive, K-fold cross-validation could be used on the training 
data. 