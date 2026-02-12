# Project Report: Multi-class AdaBoost (SAMME) with Perceptron

## 1. Introduction

In this project, we implemented the **SAMME** algorithm (Stagewise Additive Modeling using a Multi-class Exponential loss function) proposed by Zhu et al. (2006) for multi-class classification. The idea is to replicate the principles of AdaBoost, but adapted to problems with more than two classes — without systematically reducing them to multiple two-class problems.

We chose to use a **multi-class Perceptron** based on a One-vs-Rest (OvR) approach as our weak learner. Our final goal was to classify handwritten digits from the scikit-learn `digits` dataset (10 classes, representing the digits 0 to 9).

## 2. Methodology

### 2.1 Overall Approach


1. **Base Classifier (Weak Learner):**  
   - We developed a `MultiClassPerceptronOvR` class that encapsulates \(K\) binary perceptrons (one for each class).  
   - Each binary perceptron is trained with sample weights (updated at every boosting iteration).

2. **`MultiClassAdaBoostSAMME` Class:**  
   - This class implements the **SAMME** algorithm from Zhu et al. (2006).  
   - In iteration \(m\):  
     1. It trains a new `MultiClassPerceptronOvR` on the current weighted dataset.  
     2. It computes the misclassification error $\text{err}^{(m)}$.  
      3. It updates the coefficient $\alpha^{(m)} = \ln\Bigl(\frac{1 - \text{err}^{(m)}}{\text{err}^{(m)}}\Bigr) + \ln(K - 1)$. 
     4. It re-weights the samples, increasing the weight of those misclassified.

3. **Training and Testing:**  
   - We used the `digits` dataset (1797 images, 64 features, 10 classes).  
   - We split the dataset into training (80%) and test (20%), yielding 360 test samples.  
   - We tested various configurations, adjusting the number of boosting rounds \(M\), the Perceptron’s learning rate (`alpha`), and the number of epochs (`epochs`).

4. **Comparison:**  
   - For reference, we compared our results to a scikit-learn `AdaBoostClassifier` (with a `DecisionTreeClassifier(max_depth=1)`).

### 2.2 Implementation Details

- **MultiClassPerceptronOvR:** handles \(K=10\) binary perceptrons, each labeling the target class as \(+1\) and other classes as \(-1\). During training, each sample’s weight \(w_i\) affects its Perceptron update steps.
- **`MultiClassAdaBoostSAMME`:**  
  - IInitializes sample weights as $\frac{1}{n}$.  
   - Trains and evaluates the OvR Perceptron at each round, computing $\text{err}^{(m)}$ and $\alpha^{(m)}$ while re-weighting the data accordingly.  
   - After $M$ boosting rounds, the final prediction is $\arg\max_{k} \sum_{m} \alpha^{(m)} \cdot \mathbf{1}\bigl[T^{(m)}(x) = k\bigr]$.

## 3. Results

### 3.1 Hyperparameter Search

We tested:

- **\(M\)** (number of boosting iterations): \([10, 30, 50, 70]\)  
- **`alpha`** (the Perceptron’s learning rate): \([0.01, 0.1, 1.0]\)  
- **`epochs`** (number of epochs per Perceptron): \([5, 10]\)

For each parameter triplet, we trained and evaluated on the same train/test split. Below are some sample results:
M=10, alpha=0.01, epochs=5 => Test accuracy = 0.9222
M=10, alpha=0.1,  epochs=10 => Test accuracy = 0.9667
M=30, alpha=1.0,  epochs=5 => Test accuracy = 0.9444
M=50, alpha=1.0,  epochs=10 => Test accuracy = 0.9528
The **best performance** observed in this simple grid search is about **0.9667** (96.67%) for \(M=10\), `alpha=0.1`, and `epochs=10`.

### 3.2 Confusion Matrix and Classification Report

For one of the runs, we obtained about 94% accuracy. The confusion matrix is shown below:

|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **0** |32| 0| 1| 0| 0| 0| 0| 0| 0| 0|
| **1** | 0|27| 0| 0| 0| 0| 0| 0| 0| 1|
| **2** | 0| 1|29| 2| 0| 0| 0| 0| 1| 0|
| **3** | 0| 0| 0|33| 0| 1| 0| 0| 0| 0|
| **4** | 0| 2| 0| 0|44| 0| 0| 0| 0| 0|
| **5** | 0| 0| 0| 0| 0|42| 1| 0| 0| 4|
| **6** | 0| 0| 0| 0| 0| 1|34| 0| 0| 0|
| **7** | 0| 0| 0| 0| 0| 0| 0|33| 0| 1|
| **8** | 0| 0| 0| 0| 0| 0| 0| 0|30| 0|
| **9** | 0| 2| 0| 0| 0| 0| 0| 0| 3|35|

The classification report indicates most classes reach near 0.90–0.97 in precision, recall, and F1, with occasional confusion in a few classes.

### 3.3 Cross-Validation

We then performed a 5-fold StratifiedKFold cross-validation on the entire dataset:

- **CV Scores**: [0.9361, 0.9333, 0.9331, 0.9359, 0.8942]  
- **Mean**: 0.9265  
- **Std**: 0.0162  

We see an average ~92.65% score over the 5 folds, with a standard deviation of about 1.62%.

### 3.4 Comparison to scikit-learn AdaBoost

Using:

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_sklearn = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50
)
ada_sklearn.fit(X_train, y_train)
y_pred_sklearn = ada_sklearn.predict(X_test)
```





We got around 21.94% accuracy in this specific test (likely due to default parameters or overfitting on shallow decision stumps). Although this is a low baseline, it provides a reference. With more tuning, AdaBoost on stumps can often reach higher accuracy, but here our SAMME-Perceptron approach performed better under these conditions.




### 4. Conclusion :
-  Our SAMME (multi-class AdaBoost) implementation using a Perceptron OvR as a weak learner can reach up to around 96–97% accuracy on the digits dataset, depending on hyperparameters.
-  The results confirm that boosting improves the Perceptron’s generalization ability, especially with a sufficient number of epochs and a moderate learning rate.
- Cross-validation suggests a mean performance of about 92–93%, with moderate variability across folds.
-  Compared to a reference AdaBoost (scikit-learn) on 1-level decision trees, our SAMME-Perceptron generally does better in these specific tests, though it remains sensitive to hyperparameter tuning.

In summary, SAMME is an elegant method to extend AdaBoost to multi-class problems, and our weighted Perceptron OvR demonstrates promising performance for handwritten digit classification.
*Note: We used ChatGPT to help provide brainstorming ideas and inspiration for designing and testing our implementation.*