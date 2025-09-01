
# The Prediction of the quality of Red Wine
**Objective**: Predicting the quality of Red Wine (e.g, Low Quality, Mid Quality and High Quality) using selected features.

**Features**
- Volatile Acidity
- Citric Acid
- Chlorides
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

**Model**: Random Forest Classifier
- Accuracy Score: 0.86
- Confusion Matrix: 
[[  1  10   0]
 [  0 293  11]
 [  0  30  20]]
- Classification Report: 
              precision    recall  f1-score   support

           0       1.00      0.09      0.17        11
           1       0.88      0.96      0.92       304
           2       0.65      0.40      0.49        50

    accuracy                           0.86       365
   macro avg       0.84      0.48      0.53       365
weighted avg       0.85      0.86      0.84       365

- Training Score: 1.00
- Test Score: 0.86

**Model**: Decision Tree
- Accuracy Score: 0.85
- Classification Report: 
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        11
           1       0.88      0.95      0.91       304
           2       0.64      0.42      0.51        50

    accuracy                           0.85       365
   macro avg       0.50      0.46      0.47       365
weighted avg       0.82      0.85      0.83       365

- Training Score: 0.84
- Test Score: 0.85

**Insights**
- Alcohol has the highest positive correlation with the quality of the Red Wine i.e, The more alcohol in the preparation process, the better the quality of the Red Wine.
- Volatile Acidity has the highest negative correlation with the quality of the Red Wine i.e, Increase in Volatile Acidity decreases the quality of the Red Wine.

**Visualization**
- HeatMap: 'HeatMap: Selected Features.png'
- Quality vs Alcohol: 'LinePlot: Quality vs Alcohol', 'ScatterPlot for quality vs alcohol.png'
- Quality Bins: 'HistPlot: Quality Bins.png'
- Features: 'PairPlot for the selected features.png'
