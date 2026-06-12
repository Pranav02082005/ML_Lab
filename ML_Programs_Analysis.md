# ML Programs Analysis: Imports and Datasets

## Section 1: Imports Used

### Question 1: q1_metrics.py
- `pandas` (as `pd`)
- `numpy` (as `np`)
- `matplotlib.pyplot` (as `plt`)
- `sklearn.model_selection.train_test_split`
- `sklearn.linear_model.LogisticRegression`
- `sklearn.metrics.confusion_matrix`
- `sklearn.metrics.accuracy_score`
- `sklearn.metrics.precision_score`
- `sklearn.metrics.recall_score`
- `sklearn.metrics.f1_score`
- `sklearn.metrics.matthews_corrcoef`
- `sklearn.metrics.roc_curve`
- `sklearn.metrics.auc`

### Question 2: q2_find_s.py
- `pandas` (as `pd`)

### Question 3: q3_regression.py
- `pandas` (as `pd`)
- `numpy` (as `np`)
- `matplotlib.pyplot` (as `plt`)
- `sklearn.model_selection.train_test_split`
- `sklearn.linear_model.LinearRegression`
- `sklearn.linear_model.LogisticRegression`
- `sklearn.metrics.mean_squared_error`
- `sklearn.metrics.accuracy_score`
- `sklearn.metrics.precision_score`
- `sklearn.metrics.recall_score`
- `sklearn.metrics.f1_score`
- `sklearn.metrics.confusion_matrix`

### Question 4: q4_id3.py
- `pandas` (as `pd`)
- `numpy` (as `np`)
- `matplotlib.pyplot` (as `plt`)
- `sklearn.tree.DecisionTreeClassifier`
- `sklearn.tree.plot_tree`
- `sklearn.preprocessing.LabelEncoder`
- `sklearn.metrics.accuracy_score`

### Question 5: q5_knn.py
- `pandas` (as `pd`)
- `numpy` (as `np`)
- `matplotlib.pyplot` (as `plt`)
- `sklearn.model_selection.train_test_split`
- `sklearn.preprocessing.MinMaxScaler`
- `sklearn.neighbors.KNeighborsClassifier`
- `sklearn.metrics.accuracy_score`

### Question 6: q6_naive_bayes.py
- `pandas` (as `pd`)
- `numpy` (as `np`)
- `sklearn.model_selection.train_test_split`
- `sklearn.naive_bayes.GaussianNB`
- `sklearn.metrics.accuracy_score`
- `sklearn.metrics.confusion_matrix`
- `sklearn.metrics.classification_report`

### Question 7: q7_kmeans.py
- `pandas` (as `pd`)
- `matplotlib.pyplot` (as `plt`)
- `sklearn.cluster.KMeans`

### Question 8: q8_hierarchical.py
- `pandas` (as `pd`)
- `matplotlib.pyplot` (as `plt`)
- `sklearn.preprocessing.StandardScaler`
- `sklearn.cluster.AgglomerativeClustering`
- `sklearn.cluster.BisectingKMeans`
- `scipy.cluster.hierarchy.dendrogram`
- `scipy.cluster.hierarchy.linkage`

### Question 9: q9_dbscan.py
- `numpy` (as `np`)
- `pandas` (as `pd`)
- `matplotlib.pyplot` (as `plt`)
- `sklearn.cluster.DBSCAN`

### Question 10: q10_pca.py
- `pandas` (as `pd`)
- `matplotlib.pyplot` (as `plt`)
- `sklearn.decomposition.PCA`
- `sklearn.preprocessing.StandardScaler`
- `sklearn.preprocessing.LabelEncoder`

### Question 11: q11_ada_xg.py
- `sklearn.datasets.load_breast_cancer`
- `sklearn.model_selection.train_test_split`
- `sklearn.ensemble.AdaBoostClassifier`
- `sklearn.metrics.accuracy_score`
- `sklearn.metrics.roc_auc_score`
- `sklearn.metrics.classification_report`
- `xgboost.XGBClassifier`
- `numpy` (as `np`)

### Question 12: q12_rf.py
- `pandas` (as `pd`)
- `numpy` (as `np`)
- `matplotlib.pyplot` (as `plt`)
- `sklearn.ensemble.RandomForestClassifier`
- `sklearn.model_selection.train_test_split`
- `sklearn.metrics.accuracy_score`
- `sklearn.metrics.precision_score`
- `sklearn.metrics.recall_score`
- `sklearn.metrics.f1_score`

---

## Section 2: Datasets Used

### Question 1: q1_metrics.py
**Dataset:** `datasets/diabetes1.csv`
- **Description:** Diabetes dataset used for binary classification with Logistic Regression
- **Target Column:** `Outcome`

### Question 2: q2_find_s.py
**Dataset:** Custom Dataset (Manual Dictionary)
- **Description:** Custom dataset with 4 attributes for the FIND-S algorithm
- **Attributes:** Sky, Temperature, Humidity, Wind
- **Target Column:** PlayTennis
- **Samples:** 4 instances

### Question 3: q3_regression.py
**Datasets:**
1. `datasets/Advertising.csv`
   - **Description:** Advertising dataset used for Linear Regression (TV → Sales)
   - **Features:** TV spending
   - **Target:** Sales

2. `datasets/diabetes1.csv`
   - **Description:** Diabetes dataset used for Logistic Regression
   - **Target Column:** Outcome

### Question 4: q4_id3.py
**Dataset:** `datasets/Play_Tennis.csv`
- **Description:** Play Tennis dataset used for ID3 Decision Tree Classification
- **Target Column:** play

### Question 5: q5_knn.py
**Dataset:** `datasets/diabetes1.csv`
- **Description:** Diabetes dataset used for K-Nearest Neighbors classification
- **Target Column:** Outcome
- **Preprocessing:** Missing values (zeros) replaced with median for physiological columns

### Question 6: q6_naive_bayes.py
**Dataset:** Custom Dataset (Programmatically Generated)
- **Description:** Synthetic weather conditions dataset for Naive Bayes classification
- **Attributes:** Temperature, Humidity, Outlook, Wind
- **Target Column:** Play (cricket)
- **Sample Size:** 200 instances
- **Label Generation:** Rule-based (Overcast weather or Normal Humidity with Weak Wind)

### Question 7: q7_kmeans.py
**Dataset:** `iris1.csv`
- **Description:** Iris dataset used for K-Means clustering
- **Features Used:** First 2 features (columns 0-1) from the iris dataset

### Question 8: q8_hierarchical.py
**Dataset:** `datasets/iris1.csv`
- **Description:** Iris dataset used for Hierarchical Clustering (AGNES and DIANA)
- **Features Used:** All 4 features
- **Clustering Methods:** AGNES (AgglomerativeClustering) and DIANA (Bisecting K-Means)

### Question 9: q9_dbscan.py
**Dataset:** `iris1.csv`
- **Description:** Iris dataset used for DBSCAN clustering
- **Features Used:** First 2 features (columns 0-1)

### Question 10: q10_pca.py
**Dataset:** `datasets/iris1.csv`
- **Description:** Iris dataset used for Principal Component Analysis
- **Target Column:** Class
- **Features:** All features except Class column

### Question 11: q11_ada_xg.py
**Dataset:** Breast Cancer Dataset (from `sklearn.datasets.load_breast_cancer()`)
- **Description:** Built-in sklearn dataset for binary classification using ensemble methods
- **Source:** sklearn.datasets module
- **Target Column:** target (binary: malignant or benign)

### Question 12: q12_rf.py
**Dataset:** `datasets/diabetes1.csv`
- **Description:** Diabetes dataset used for Random Forest classification
- **Target Column:** Outcome
- **Analysis:** Effect of number of trees (1, 5, 10, 20, 50, 100, 150, 200) on performance metrics
