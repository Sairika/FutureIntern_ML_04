# Credit Card Fraud Detection

In this project, a model is developed to identify fraudulent transactions within a highly imbalanced dataset, where fraudulent activities are significantly less common than legitimate transactions. Various strategies are implemented to address this imbalance, including Random UnderSampling and oversampling techniques like SMOTE (Synthetic Minority Over-sampling Technique). Different classifiers, such as Logistic Regression, Random Forest, SVC, KNN and Neural Networks, are applied and evaluated to determine their effectiveness in detecting fraud. Throughout the analysis, precision, recall, and F1-score are prioritized as key metrics to ensure that the models are not only accurate but also capable of identifying fraud while minimizing false positives. This study explores the preprocessing steps, model training, and the challenges associated with imbalanced data, ultimately providing insights into best practices for fraud detection in financial transactions.


## Table of Contents

- [Objective](#objective)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Handling Imbalanced Data](#handling-imbalanced-data)
- [Feature Scaling](#feature-scaling)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [GridSearchCV](#gridsearchcv)
- [Learning Curve Analysis](#learning-curve-analysis)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)

## Objective

- To implement strategies that manage the imbalanced dataset effectively.
- To evaluate and apply various classifiers, including Logistic Regression and Neural Networks.
- To train models on processed data in order to recognize fraud patterns.
- To use precision, recall, and F1-score as key metrics for model evaluation.


## Dataset

The dataset is provided and cleaned, with no missing or null values.
- **Class Imbalance:** The dataset is highly imbalanced, with non-fraudulent transactions constituting 99.83% and fraudulent ones only 0.17%.
- **Transaction Amount:** The average transaction amount is approximately USD 88.
- **Features:** The dataset contains features like 'Time', 'Amount', and anonymized features 'V1' to 'V28'.

## Data Preprocessing

- **Class Distribution Analysis:** Analyzed the class (target variable) distribution to understand the extent of imbalance.
- **Feature Scaling:** Applied `RobustScaler` to the 'Time' and 'Amount' features to handle outliers and bring them on a comparable scale.

## Exploratory Data Analysis (EDA)

- **Correlation Matrices:** Used to understand which features have significant positive or negative correlations with fraudulent transactions.
- **Boxplots:** Visualized the distribution of features across fraudulent and non-fraudulent transactions, focusing on features with strong correlations to the target class.

## Handling Imbalanced Data

- Both undersampling and oversampling (SMOTE) techniques were explored. SMOTE consistently outperformed undersampling, providing a better balance between precision and recall for fraud detection across different models.

## Feature Scaling

- Applied `RobustScaler` to the 'Time' and 'Amount' features, which effectively handles outliers by focusing on the interquartile range.

## Dimensionality Reduction

- **t-SNE:** Used to visualize high-dimensional data and uncover hidden patterns.
- **Truncated SVD and PCA:** Applied for dimensionality reduction, especially useful for sparse and high-dimensional data.

## Model Training and Evaluation

- Trained several classifiers, including:
  - **Logistic Regression**
  - **K-Nearest Neighbors (KNN)**
  - **Support Vector Classifier (SVC)**
  - **Random Forest**
  - **Neural Netwrok**
- **Metrics Used:** Precision, Recall, F1-score, and AUC were the key metrics used for model evaluation.


## Results

- Traditional Classifiers (undersampling within cross-validation):
     * Logistic Regression: AUC 0.9696
     * Support Vector Classifier: AUC 0.9619
     * K-Nearest Neighbors: AUC 0.9343
     * Random Forest: AUC 0.9305
   
- Logistic Regression (SMOTE):
     * Accuracy: 0.9859
     * Precision-Recall Score: 0.70

- Neural Network (SMOTE):
     * Accuracy: 1.00
     * Precision for fraud: 0.80
     * Recall for fraud: 0.70
     * F1-score for fraud: 0.75
- Best performing model: The neural network with SMOTE emerged as the most promising approach, showing high accuracy and a good balance between precision and recall in fraud detection.
     
## Conclusion

- The neural network model, enhanced with SMOTE, was identified as the top performer, achieving an impressive accuracy of 100% and an F1-score of 0.75 for fraud detection.
- SMOTE consistently outperformed undersampling across different models, highlighting its effectiveness in handling imbalanced datasets.
- Proper implementation of sampling techniques within cross-validation proved crucial for obtaining realistic and reliable performance estimates.
- However, it is important to note that the neural network trained on the oversampled dataset sometimes performed worse in correctly predicting fraud transactions compared to the model trained on the undersampled dataset. One reason for this discrepancy could be that outlier removal was applied only to the random undersampled dataset and not to the oversampled one. In the undersampled dataset, our model struggled to correctly classify many non-fraud transactions, misclassifying them as fraud cases. This issue highlights a significant risk: if a model incorrectly labels legitimate transactions as fraud, it could lead to customer dissatisfaction and an increase in complaints due to blocked cards for regular purchases.
- Future work could explore more advanced neural network architectures, ensemble methods, or the integration of additional data sources to further improve fraud detection capabilities.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Reference

This project is inspired by methodologies from the following Kaggle notebook:
- Janiobachmann, "Credit Fraud Dealing with Imbalanced Datasets", Kaggle. Available at: https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets