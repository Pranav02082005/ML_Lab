# ML Programs Viva: Questions and Answers

## Question 1: Metrics and Model Evaluation (q1_metrics.py)

**Q1.1: What is a confusion matrix and what are its components?**

A confusion matrix is a table used to describe performance of a classification model. It has 4 components:
- **TP (True Positives):** Correctly predicted positive cases
- **TN (True Negatives):** Correctly predicted negative cases
- **FP (False Positives):** Negative cases predicted as positive (Type I error)
- **FN (False Negatives):** Positive cases predicted as negative (Type II error)

**Q1.2: Define Accuracy, Precision, and Recall. When would you prefer Recall over Precision?**

- **Accuracy:** (TP + TN) / (TP + TN + FP + FN) - Overall correctness of predictions
- **Precision:** TP / (TP + FP) - Of all positive predictions, how many were correct
- **Recall:** TP / (TP + FN) - Of all actual positives, how many were identified

Prefer Recall when false negatives are costly (e.g., disease diagnosis, fraud detection). You want to catch all positive cases even if it means more false alarms.

**Q1.3: What is F1-Score and why is it useful?**

F1-Score = 2 × (Precision × Recall) / (Precision + Recall). It is the harmonic mean of precision and recall. It's useful when you want a single metric that balances both precision and recall, especially when classes are imbalanced.

**Q1.4: Explain Matthews Correlation Coefficient (MCC).**

MCC = (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]. It's a correlation coefficient between actual and predicted classifications. MCC ranges from -1 to +1, where +1 is perfect prediction, 0 is random, and -1 is inverse prediction. It's especially useful for imbalanced datasets.

**Q1.5: What is ROC curve and AUC? What does AUC represent?**

ROC (Receiver Operating Characteristic) curve plots True Positive Rate (TPR/Recall) vs False Positive Rate (FPR) at different classification thresholds. AUC (Area Under Curve) ranges from 0 to 1, representing the probability that the model ranks a random positive example higher than a random negative example. AUC = 0.5 means random classifier, AUC = 1.0 means perfect classifier.

---

## Question 2: FIND-S Algorithm (q2_find_s.py)

**Q2.1: What is the FIND-S algorithm and what is its purpose?**

FIND-S (Find Simplest hypothesis) is a concept learning algorithm from the version space approach. It finds the most specific hypothesis consistent with positive training examples. Starting from the most specific hypothesis (first positive example), it generalizes by replacing differing attribute values with "?" until it covers all positive examples.

**Q2.2: What are the limitations of FIND-S?**

1. Cannot handle noisy data - fails if positive examples conflict
2. Doesn't identify multiple consistent hypotheses
3. Cannot handle negative examples
4. Requires instances to match the predefined attribute-value format
5. Outputs a single hypothesis without indicating confidence

**Q2.3: How does FIND-S handle attribute generalization?**

When processing positive examples sequentially, if an attribute value in the current hypothesis differs from the value in the new positive example, it's generalized to "?" (meaning "any value" is acceptable). This makes the hypothesis less specific but consistent with all seen positive examples.

**Q2.4: Compare FIND-S with other concept learning methods.**

FIND-S finds the most specific hypothesis. Candidate Elimination finds all consistent hypotheses. FIND-S is more efficient but less informative. Decision Trees can handle complex patterns but require more data. FIND-S is simpler but limited to version space framework.

---

## Question 3: Regression (q3_regression.py)

**Q3.1: What is Linear Regression and what does the equation Y = mX + b represent?**

Linear Regression models the linear relationship between independent variable(s) and dependent variable. The equation Y = mX + b represents a straight line where m is the slope (rate of change) and b is the y-intercept. It minimizes the sum of squared residuals using the least squares method.

**Q3.2: What is the difference between Linear Regression and Logistic Regression?**

- **Linear Regression:** Used for continuous output values. Uses linear equation. Output range: -∞ to +∞
- **Logistic Regression:** Used for binary classification (0/1). Uses sigmoid function. Output range: 0 to 1 (probability). Despite "Regression" in name, it's a classification algorithm.

**Q3.3: What is the sigmoid function and why is it used in Logistic Regression?**

Sigmoid function: σ(z) = 1 / (1 + e^(-z)). It maps any input to a value between 0 and 1, making it perfect for probability representation. It's smooth, differentiable, and naturally interpretable as probability of the positive class.

**Q3.4: How do you evaluate Linear Regression models?**

Common metrics:
- **MSE (Mean Squared Error):** Average of squared residuals. Penalizes large errors.
- **RMSE (Root MSE):** Square root of MSE. Same units as target variable.
- **R² Score:** Proportion of variance explained. Ranges 0-1, higher is better.
- **MAE:** Mean absolute error. Less sensitive to outliers than MSE.

**Q3.5: What is overfitting in regression and how to prevent it?**

Overfitting occurs when model learns training data too well, including noise, and performs poorly on new data. Prevention methods: regularization (L1/L2), cross-validation, simpler models, more training data, feature selection.

---

## Question 4: ID3 Decision Trees (q4_id3.py)

**Q4.1: What is ID3 (Iterative Dichotomiser 3) algorithm?**

ID3 is a top-down, greedy decision tree algorithm. It recursively splits data by selecting attributes that best separate classes. It uses **Entropy** and **Information Gain** to choose splitting criteria. It builds trees suitable for categorical features and stops when all instances belong to one class.

**Q4.2: Explain Information Gain and why it's used for attribute selection.**

Information Gain (IG) = Parent Entropy - (weighted average of children entropies). It measures reduction in uncertainty after splitting. ID3 selects the attribute with highest Information Gain because it best reduces disorder, leading to purer splits and simpler trees.

**Q4.3: What is Entropy and how is it calculated?**

Entropy measures disorder/impurity in a dataset: H(S) = -Σ(p_i × log₂(p_i)) where p_i is proportion of class i. Entropy = 0 when all instances belong to one class (pure). Entropy = 1 when classes are perfectly balanced (maximum impurity).

**Q4.4: What are limitations of ID3?**

1. Biased toward attributes with more values
2. Doesn't handle continuous attributes (requires discretization)
3. Cannot handle missing values well
4. Prone to overfitting (builds complete trees)
5. Greedy approach - may not find globally optimal tree

**Q4.5: How does decision tree handle categorical and continuous features?**

For categorical features: Direct branching on attribute values. For continuous features: Must be discretized (binned) into categories first or uses threshold-based splits (only in C4.5/C5.0, not pure ID3).

---

## Question 5: K-Nearest Neighbors (q5_knn.py)

**Q5.1: How does KNN algorithm work?**

KNN is a lazy learning algorithm. For a query point: 
1. Calculate distances to all training points
2. Find k nearest neighbors
3. For classification: majority voting among neighbors
4. For regression: average of neighbor values
No explicit model is built; classification happens at prediction time.

**Q5.2: What distance metrics can be used in KNN? Why is scaling important?**

Common metrics:
- **Euclidean:** √(Σ(x_i - y_i)²) - most common
- **Manhattan:** Σ|x_i - y_i| - sum of absolute differences
- **Cosine:** measures angle similarity
- **Hamming:** for categorical data

Scaling is crucial because without it, features with large ranges dominate distance calculations, making small-range features irrelevant.

**Q5.3: How do you choose the optimal value of k?**

Use cross-validation: try different k values and measure performance. Generally:
- Small k (1-3): low bias, high variance (overfitting)
- Large k: high bias, low variance (underfitting)
- Odd k for binary classification avoids ties
- Typically k = √n or use validation set

**Q5.4: What are advantages and disadvantages of KNN?**

**Advantages:** Simple, no training phase, works for multi-class problems, no assumptions about data distribution
**Disadvantages:** Slow at prediction (distance to all points), sensitive to irrelevant features, poor performance on high-dimensional data, memory intensive, requires scaling

**Q5.5: Why is KNN called a "lazy" learning algorithm?**

Because it doesn't build an explicit model during training - all computation happens at prediction time. It simply stores training data and defers learning until a query arrives, making training fast but prediction slow.

---

## Question 6: Naive Bayes (q6_naive_bayes.py)

**Q6.1: What is Naive Bayes and what is the "naive" assumption?**

Naive Bayes is a probabilistic classifier based on Bayes' theorem: P(Class|Features) = P(Features|Class) × P(Class) / P(Features). The "naive" assumption assumes all features are conditionally independent given the class label. This simplification makes computation tractable though the assumption rarely holds in reality.

**Q6.2: Write and explain Bayes' theorem.**

P(A|B) = P(B|A) × P(A) / P(B), where:
- P(A|B) = Posterior probability (what we want)
- P(B|A) = Likelihood (probability of evidence given class)
- P(A) = Prior probability (class probability before evidence)
- P(B) = Evidence (total probability of evidence)

**Q6.3: What are different variants of Naive Bayes?**

1. **Gaussian NB:** Assumes features follow normal distribution. Best for continuous data.
2. **Multinomial NB:** Counts features. Used for document classification, text mining.
3. **Bernoulli NB:** Features are binary (present/absent). Used for binary text classification.

**Q6.4: What are advantages and disadvantages of Naive Bayes?**

**Advantages:** Fast training/prediction, works well with small datasets, handles high dimensions, probabilistic output, works with both continuous and categorical data
**Disadvantages:** Independence assumption rarely holds, sensitive to feature scaling, zero-frequency problem (handled via smoothing), assumes prior class probability

**Q6.5: How does Naive Bayes handle the zero-frequency problem?**

When a feature-class combination never appears in training data, probability becomes 0, zeroing out entire prediction. Laplace smoothing adds 1 to all counts: P(feature|class) = (count + 1) / (total + num_features), ensuring no zero probabilities.

---

## Question 7: K-Means Clustering (q7_kmeans.py)

**Q7.1: How does K-Means algorithm work? Describe the steps.**

K-Means iterative algorithm:
1. Initialize k cluster centers randomly
2. Assign each point to nearest center (Euclidean distance)
3. Update centers as mean of assigned points
4. Repeat steps 2-3 until convergence (centers don't change or max iterations reached)

It's an unsupervised method that partitions data into k non-overlapping clusters.

**Q7.2: What is the Elbow Method and how does it help choose k?**

Elbow Method plots WCSS (Within-Cluster Sum of Squares) vs k. WCSS decreases as k increases. The "elbow point" (where decrease slows significantly) suggests optimal k. It helps identify k beyond which adding clusters yields diminishing improvements.

**Q7.3: What are advantages and limitations of K-Means?**

**Advantages:** Simple, fast, scalable, works well for spherical clusters, produces hard clusters
**Limitations:** Must specify k in advance, sensitive to initial centroids, may converge to local minima, assumes equal-sized clusters, struggles with non-convex shapes, sensitive to outliers

**Q7.4: How does initial centroid selection affect K-Means?**

Poor initialization can lead to suboptimal solutions (local minima). K-Means++ improves this by probabilistically selecting initial centroids far apart, reducing chance of poor local optima and improving convergence.

**Q7.5: How would you handle categorical data in K-Means?**

K-Means is designed for continuous data. For categorical data:
- Use K-Modes algorithm (replaces means with modes)
- Encode categories numerically (ordinal/one-hot encoding) with caution
- Use distance metrics suitable for categorical data
- Use specialized algorithms like hierarchical clustering

---

## Question 8: Hierarchical Clustering (q8_hierarchical.py)

**Q8.1: What is Hierarchical Clustering? Distinguish between AGNES and DIANA.**

Hierarchical Clustering builds a tree-like hierarchy of clusters.
- **AGNES (Agglomerative):** Bottom-up approach. Starts with each point as cluster, merges similar clusters iteratively.
- **DIANA (Divisive):** Top-down approach. Starts with all points in one cluster, recursively splits into smaller clusters.

AGNES is more common and efficient; DIANA is computationally expensive.

**Q8.2: What linkage methods are used in Hierarchical Clustering?**

1. **Single Linkage:** Minimum distance between clusters. Sensitive to outliers.
2. **Complete Linkage:** Maximum distance between clusters. More conservative.
3. **Average Linkage:** Average distance between clusters. Balanced approach.
4. **Ward Linkage:** Minimizes within-cluster variance. Often gives better results.

**Q8.3: What is a Dendrogram and how do you interpret it?**

Dendrogram is a tree diagram showing hierarchical cluster merges. Y-axis shows distance at which clusters merge. Horizontal cuts at different heights produce different numbers of clusters. Taller branches indicate more dissimilar clusters being merged.

**Q8.4: Compare Hierarchical Clustering with K-Means.**

- **Hierarchical:** No need to specify cluster count, produces hierarchy, deterministic (for AGNES)
- **K-Means:** Must specify k, partitional (non-hierarchical), non-deterministic (depends on initialization), faster

Hierarchical is better when you want to explore multiple granularity levels; K-Means better for speed and fixed cluster needs.

**Q8.5: What are advantages and disadvantages of Hierarchical Clustering?**

**Advantages:** No need to specify k, produces dendrogram showing relationships, deterministic (AGNES), handles non-spherical clusters
**Disadvantages:** Computationally expensive (O(n²)), sensitive to noise/outliers, hard to reverse decisions once merged, high memory usage, linkage choice affects results

---

## Question 9: DBSCAN (q9_dbscan.py)

**Q9.1: How does DBSCAN algorithm work? Explain eps and min_samples parameters.**

DBSCAN (Density-Based Spatial Clustering) groups points based on density:
- **eps:** Radius around each point. Points within eps are neighbors.
- **min_samples:** Minimum neighbors required to form cluster.

Process: Start with unvisited point, if it has ≥ min_samples neighbors within eps, create cluster and recursively add reachable neighbors. Otherwise, mark as noise/outlier.

**Q9.2: What are Core Points, Border Points, and Noise Points?**

- **Core Points:** Have ≥ min_samples neighbors within eps. Guaranteed in some cluster.
- **Border Points:** Not core but within eps of core point. In cluster but on boundary.
- **Noise Points:** Fewer than min_samples neighbors. Often treated as outliers.

**Q9.3: What are advantages of DBSCAN over K-Means?**

1. No need to specify number of clusters
2. Finds arbitrary-shaped clusters (non-spherical)
3. Automatically identifies outliers/noise points
4. Works well with varying cluster densities
5. Density-based approach (realistic for many datasets)

**Q9.4: What are disadvantages and challenges of DBSCAN?**

1. Sensitive to eps and min_samples parameters
2. Difficult to choose optimal parameters (no standard method)
3. Struggles with varying density clusters (different eps needed)
4. High computational complexity (O(n²)) without spatial indexing
5. Doesn't assign cluster membership probabilities

**Q9.5: How do you choose eps and min_samples parameters?**

**For eps:** Use k-distance graph (plot distances to k-th nearest neighbor, sorted). Look for "knee" point - represents transition from dense to sparse regions. Set eps at knee.
**For min_samples:** Typically set to dimensionality (d) or 2×d. Can use cross-validation.

---

## Question 10: PCA (q10_pca.py)

**Q10.1: What is Principal Component Analysis (PCA) and what does it do?**

PCA is unsupervised dimensionality reduction technique. It transforms high-dimensional data into lower-dimensional space while preserving maximum variance. It finds principal components (orthogonal directions of maximum variance) and projects data onto them.

**Q10.2: Explain the concept of Principal Components.**

Principal Components are new orthogonal axes (eigenvectors of covariance matrix) ordered by variance they capture. PC1 captures most variance, PC2 captures second most (uncorrelated with PC1), etc. They form new coordinate system where data is uncorrelated.

**Q10.3: Why is standardization important before applying PCA?**

Features must be standardized (mean=0, std=1) because PCA is scale-sensitive. Without standardization, features with large ranges dominate PCA, and components depend on measurement units. Standardization ensures all features contribute equally.

**Q10.4: How do you interpret Explained Variance Ratio?**

Explained Variance Ratio shows proportion of total variance captured by each PC. For example, if first two PCs have ratios 0.72 and 0.18, they capture 90% of total variance. Sum of all ratios equals 1. Higher ratio means more information retained.

**Q10.5: What are advantages and disadvantages of PCA?**

**Advantages:** Reduces dimensions/computation, removes correlation, improves visualization, helps prevent overfitting, faster training
**Disadvantages:** Components are hard to interpret (linear combinations), assumes linear relationships, sensitive to outliers, requires standardization, loses some information

**Q10.6: When should you use PCA?**

Use PCA when: dealing with high-dimensional data, computational efficiency needed, visualizing high-dim data, features are correlated, removing noise, want unsupervised feature extraction. Don't use when: interpretability crucial, relationships non-linear, need supervised feature selection.

---

## Question 11: Ensemble Methods - AdaBoost and XGBoost (q11_ada_xg.py)

**Q11.1: What is ensemble learning? Why combine multiple models?**

Ensemble learning combines predictions from multiple models to improve overall performance. Combining different models:
- Reduces overfitting (averaging reduces variance)
- Increases robustness (diversity in predictions)
- Improves generalization
- Can leverage strengths of different algorithms
- Often beats single best model (Wisdom of Crowds)

**Q11.2: How does AdaBoost work?**

AdaBoost (Adaptive Boosting) sequentially builds weak learners:
1. Initialize equal weights to all samples
2. Train weak learner, calculate error rate
3. Increase weights for misclassified samples
4. Repeat: each new learner focuses on previously hard examples
5. Final prediction: weighted majority vote of all weak learners

Emphasizes hard examples, improving on weak performance areas.

**Q11.3: What is the difference between Bagging and Boosting?**

- **Bagging (Bootstrap Aggregating):** Trains models on random data subsets independently. Reduces variance. Examples: Random Forest, Isolation Forest.
- **Boosting:** Sequentially trains models, each focusing on previous mistakes. Reduces bias. Examples: AdaBoost, Gradient Boosting, XGBoost.

Boosting typically achieves better accuracy; Bagging is simpler and parallelizable.

**Q11.4: How does XGBoost improve upon Gradient Boosting?**

XGBoost optimizations:
1. Regularization (L1/L2) prevents overfitting
2. Parallel processing for speed
3. Handles missing values automatically
4. Smart tree pruning (depth-first with score pruning)
5. Weighted quantile sketch for approximate splits
6. Out-of-core computation for large datasets
7. Better default hyperparameters

Results in faster training, better performance, handles sparse data.

**Q11.5: What is gradient boosting and how does it relate to XGBoost?**

Gradient Boosting builds trees sequentially, each correcting previous trees' residuals. Loss gradient guides where to improve. New tree's predictions are weighted and added to ensemble. XGBoost is optimized implementation with regularization, parallelization, and pruning improvements over standard gradient boosting.

**Q11.6: What hyperparameters are important in AdaBoost and XGBoost?**

**AdaBoost:** n_estimators (weak learners), learning_rate, base_estimator choice
**XGBoost:** n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, lambda (L2 regularization), alpha (L1 regularization), gamma (minimum loss reduction)

---

## Question 12: Random Forest (q12_rf.py)

**Q12.1: What is Random Forest and how does it work?**

Random Forest is ensemble of decision trees:
1. Build multiple decision trees on random subsets of data (bagging)
2. At each split, consider random feature subset (adds randomness)
3. Train trees fully (no pruning)
4. For classification: majority voting
5. For regression: average predictions

Randomness in data and features prevents overfitting and improves generalization.

**Q12.2: What is the difference between Random Forest and Decision Tree?**

- **Decision Tree:** Single tree, prone to overfitting, high variance, interpretable
- **Random Forest:** Multiple trees, reduced overfitting, lower variance, higher bias, less interpretable but better predictions

Random Forest trades interpretability for accuracy and robustness.

**Q12.3: What is "Out-of-Bag" (OOB) error and how is it useful?**

OOB error uses samples not in bootstrap sample for validation (about 1/3). Provides unbiased performance estimate without separate test set. Useful for:
- Cross-validation without extra computation
- Feature importance estimation
- Parameter tuning
- Detecting overfitting

**Q12.4: How does Random Forest measure feature importance?**

Feature importance in Random Forest:
- **Gini Importance:** Based on impurity decrease across all splits using that feature
- **Permutation Importance:** Measure accuracy drop when feature values are randomly shuffled
- Higher importance means feature is more useful for splits

Helps identify which features drive predictions.

**Q12.5: What are advantages and disadvantages of Random Forest?**

**Advantages:** Reduces overfitting, handles non-linear relationships, captures feature interactions, produces feature importance, robust to outliers, parallelizable, minimal tuning needed
**Disadvantages:** Memory intensive, slower predictions (many trees), less interpretable, biased toward high-cardinality features, can struggle with linear patterns, requires more data than single tree

**Q12.6: How does increasing number of trees affect Random Forest performance?**

More trees generally improve:
- Training accuracy (captures more patterns)
- Test accuracy (reduces variance via averaging)
- Stability of predictions
- Feature importance estimates

However, diminishing returns appear. Computational cost increases linearly. Optimal number found via validation curve (as shown in q12_rf.py).

---

## General Concepts and Comparisons

**Q13.1: Compare supervised vs unsupervised learning.**

| Aspect | Supervised | Unsupervised |
|--------|-----------|-------------|
| Labels | Required | Not required |
| Goal | Predict target | Find patterns/structure |
| Examples | Classification, Regression | Clustering, Dimensionality Reduction |
| Evaluation | Easy (compare to labels) | Hard (no ground truth) |
| Use cases | Medical diagnosis, fraud detection | Customer segmentation, anomaly detection |

**Q13.2: Compare Classification vs Clustering.**

- **Classification:** Supervised, predicts discrete classes, requires labeled data, models learned from examples, evaluation straightforward
- **Clustering:** Unsupervised, groups similar items, no labels needed, patterns discovered from data, evaluation subjective

**Q13.3: What are the main challenges in Machine Learning?**

1. **Data Quality:** Missing values, outliers, imbalanced classes
2. **Overfitting:** Model too specific to training data
3. **Underfitting:** Model too simple, misses patterns
4. **Scalability:** Computational and memory constraints
5. **Feature Engineering:** Creating relevant features
6. **Hyperparameter Tuning:** Finding optimal parameters
7. **Class Imbalance:** Unequal class distribution
8. **Interpretability:** Understanding model decisions

**Q13.4: What is cross-validation and why is it important?**

Cross-validation splits data into k folds, trains k models (each on k-1 folds), and evaluates on remaining fold. Average performance estimates true generalization.

Importance:
- Uses all data for both training and validation
- Reduces variance in performance estimates
- Detects overfitting
- Works with small datasets
- More reliable than single train-test split

**Q13.5: When should you use different types of models?**

- **Linear Models:** Fast, interpretable, linear relationships, small data
- **Tree-based:** Non-linear patterns, feature interactions, handles categorical data
- **Ensemble:** Best accuracy, complex patterns, robust
- **Neural Networks:** Very complex patterns, huge data, computational resources available
- **Clustering:** Exploratory analysis, customer segmentation, anomaly detection

Choose based on: data size, complexity, interpretability needs, computational resources, problem type.
