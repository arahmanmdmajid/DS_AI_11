# Admission Predict Analysis

This project aims to predict the chance of admission for a student based on various academic factors.

## Dataset

The dataset used is `Admission_Predict.csv`, which contains features like GRE Score, TOEFL Score, University Rating, SOP (Statement of Purpose), LOR (Letter of Recommendation), CGPA, Research, and the predicted variable 'Chance of Admit'.

## Approach

1. The dataset was loaded, and initial checks were performed for inconsistencies, missing values, and duplicates.
2. Performed the following data preprocessing steps:
    *   Drop the 'Serial No.' column
    *   Check for outliers using box plots.
    *   Visualize distribution using histograms.
    *   Analyze the relationships between features using scatter plots.
    *   Also checked the correlation using a heatmap.
    *   Scale the features using `MinMaxScaler`
3. Trained the following models
    *   Linear Regression
    *   Decision Tree Regressor
    *   Random Forest Regressor
4.  Evaluate the performance of each model by calculating the R² score.
5.  Performed hyperparameter tuning
6. Performed Feature Selection by selecting the top 5 most important features and retraining the models again.

## Results 

### Model Performance (Before Feature Selection)

| Model                      | R² Score        | 
| :------------------------- | :-------------- | 
| Linear Regression          | 0.823           | 
| Decision Tree (Untuned)    | 0.692           | 
| Decision Tree (Tuned)      | 0.828           | 
| Random Forest (Untuned)    | 0.874           | 
| Random Forest (Tuned)      | 0.874           | 

### Model Performance (With Top 5 Features)

After selecting the top 5 features, the models were retrained:

| Model                      | R² Score        |
| :------------------------- | :-------------- |
| Linear Regression          | 0.823           |
| Decision Tree (Tuned)      | 0.828           |
| Random Forest (Tuned)      | 0.874           |

The R² scores for the models trained on the top 5 features are very similar to the scores obtained with all features. This suggests that the dropped features had minimal impact on the model's performance.

## Conclusion

Based on this analysis, the Random Forest Regressor appears to be the most effective model for predicting the chance of admission. 
