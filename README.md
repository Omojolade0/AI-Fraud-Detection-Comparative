# AI-Fraud-Detection-Comparative-Study
Detect credit card fraud using machine learning algorithms like KNN, Random Forest, and SVM

 **DataSet**
https://drive.google.com/file/d/1kQIsLCaxJzk_3seZyuSTEnYoJU4l6etW/view?usp=sharing

**Implementation:**

**Data Preprocessing:**
Handling missing values, duplicates, and feature encoding.
Scaling numerical features and extracting transaction hour information.

**Model Training and Evaluation:**
Utilizing SMOTE for class imbalance.
Evaluating model performance with accuracy metrics and confusion matrices.

**Implemented Algorithms:**
K-Nearest Neighbors (KNN)
Support Vector Machines (SVM)
Random Forest

**Results:**
KNN, Random Forest, SVM:
Algorithm  Portion  Accuracy  Timeframe
KNN        0.1      0.99      4 Seconds
           1        0.95      57 Seconds
RF         0.1      0.99      51 Seconds
           1        0.99      19 Minutes
SVM        0.05     0.992     42 Seconds
           0.1      0.993     2 Minutes
           0.2      0.994     5 Minutes
           0.5      0.994     >30 Minutes
           1        Nil       >4 Hours

**Future Work:**
Explore additional algorithms and hyperparameter tuning.
Investigate advanced feature engineering techniques.
Enhance scalability and efficiency for larger datasets.

**How to Run:**
Clone the repository.
Install Python and required libraries.
Download the Kaggle dataset and place it in the project directory.
Run the Python script (Project.py) to execute the code.
