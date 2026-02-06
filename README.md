   # Credit Card Fraud Detection

   ## Overview

   This project implements a machine learning-based credit card fraud detection system using Python. The model analyzes transaction data to identify potentially fraudulent activities, helping financial institutions prevent unauthorized transactions.

   ## Features

   - Data preprocessing and feature Engineering
   - Categorical variable encoding
   - Machine learning model training using LightGBM
   - Evaluation metrics including ROC-AUC, classification report, and confusion matrix
   - Handling imbalanced datasets with SMOTE oversampling

   ## Dataset

   The project uses a credit card transaction dataset (`dataset.csv`) containing transaction details such as:
   - Transaction amount
   - Merchant information
   - Customer demographics
   - Transaction timestamps
   - Fraud labels

   **Note:** The dataset is not included in this repository due to privacy and size constraints. You will need to obtain or generate your own dataset with similar structure.

   ## Requirements

   - Python 3.7+
   - Jupyter Notebook
   - Required Python packages:
   - pandas
   - numpy
   - lightgbm
   - seaborn
   - matplotlib
   - scikit-learn
   - imbalanced-learn
   - geopy
   - joblib

   ## Installation

   1. Clone this repository:
      ```bash
      git clone https://github.com/yourusername/credit-card-fraud-detection.git
      cd credit-card-fraud-detection
      ```

   2. Install the required packages:
      ```bash
      pip install pandas numpy lightgbm seaborn matplotlib scikit-learn imbalanced-learn geopy joblib
      ```

   3. Place your dataset file as `dataset.csv` in the project root directory.

   ## Usage

   ### Running the Jupyter Notebook

   1. Start Jupyter Notebook:
      ```bash
      jupyter notebook
      ```

   2. Open `app.ipynb` and run the cells sequentially to:
      - Load and explore the dataset
      - Preprocess the data (date conversion, feature engineering)
      - Train the fraud detection model
      - Evaluate model performance

   ### Python Script (app.py)

   The `app.py` file is currently empty but can be used to create a standalone script version of the notebook for production deployment.

   ## Model Training Process

   1. **Data Loading**: Load transaction data from CSV
   2. **Feature Engineering**: Extract time-based features (hour, day, month)
   3. **Data Cleaning**: Remove unnecessary columns
   4. **Encoding**: Convert categorical variables to numerical using LabelEncoder
   5. **Handling Imbalance**: Apply SMOTE to balance the dataset
   6. **Model Training**: Train LightGBM classifier
   7. **Evaluation**: Assess model performance using various metrics

   ## Evaluation Metrics

   The model is evaluated using:
   - ROC-AUC Score
   - Classification Report (Precision, Recall, F1-Score)
   - Confusion Matrix
   - ROC Curve

   ## Contributing

   Contributions are welcome! Please feel free to submit a Pull Request.

   ## License

   This project is licensed under the MIT License - see the LICENSE file for details.

   ## Disclaimer

   This is a demonstration project for educational purposes. In a real-world scenario, fraud detection systems require extensive validation, regulatory compliance, and integration with secure data pipelines.