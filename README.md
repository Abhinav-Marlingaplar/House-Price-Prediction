# House Price Prediction using XGBoost Regressor

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%2CNumpy%2CSklearn%2CMatplotlib%2CSeaborn%2CXGBoost%2CRandomizedSearchCV-brightgreen.svg)

## Overview

This project aims to develop an accurate predictive model for house prices, primarily focusing on a regression task from a Kaggle competition. By leveraging a comprehensive dataset of residential properties and various influential features, the goal is to build a robust system capable of estimating property values. The project extensively utilizes the XGBoost Regressor, known for its performance in structured data, and includes steps for thorough data understanding and model optimization.

## Dataset

The project utilizes the "House Prices - Advanced Regression Techniques" dataset, commonly found on Kaggle. It includes `train.csv`, `test.csv`, and `data_description.txt`, providing detailed information about residential properties in Ames, Iowa, and their respective sale prices.

## Key Features

* **Extensive Data Preprocessing:** Comprehensive handling of missing values, outlier detection (through skewness analysis), and feature engineering to prepare the data for modeling.
* **Robust XGBoost Regressor:** Implementation of the powerful XGBoost algorithm, optimized using hyperparameter tuning (e.g., RandomizedSearchCV) to achieve high predictive accuracy.
* **Performance Evaluation:** Thorough assessment of the model's predictive capabilities using standard regression metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2).
* **Kaggle Competition Submission:** The model's performance is validated on an unseen test dataset, with predictions formatted for submission to the Kaggle "House Prices - Advanced Regression Techniques" competition.

## Technologies Used

* **Python:** The core programming language for the entire project.
* **Pandas:** Essential for efficient data manipulation and analysis.
* **NumPy:** For numerical operations, especially array manipulations.
* **Matplotlib and Seaborn:** For creating insightful data visualizations and exploratory data analysis.
* **Scikit-learn (sklearn):** Used for data splitting, preprocessing utilities, and evaluation metrics.
* **XGBoost:** The primary machine learning library for implementing the gradient boosting model.

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/Abhinav-Marlingaplar/House-Price-Prediction.git](https://github.com/Abhinav-Marlingaplar/House-Price-Prediction.git)
    cd House-Price-Prediction
    ```

2.  **Install required libraries:**

    This project relies on the following Python libraries. You can install them using pip, ideally within a virtual environment:

    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn xgboost
    # Or, if you create a requirements.txt:
    # pip install -r requirements.txt
    ```

## Usage

1.  **Data Acquisition:** Ensure the `train.csv`, `test.csv`, and `data_description.txt` files are present in the project's root directory. These files are typically downloaded from the Kaggle competition page. *(Note: These files are included in this repository for convenience).*

2.  **Run the Jupyter Notebook:**
    The core analysis and model training are performed within the Jupyter Notebook.

    ```bash
    jupyter notebook House_Price_Prediction.ipynb
    ```
    Follow the steps within the notebook to load data, perform EDA, preprocess features, train the XGBoost model, evaluate its performance, and generate the `submission.csv` file.

3.  **Generate Submission File:**
    After running the notebook, a `submission.csv` file will be generated in the project directory, ready for submission to the Kaggle competition.

## Results

The XGBoost model demonstrated strong predictive capabilities on the held-out test set and achieved a competitive score in the Kaggle competition.

**Test Set Performance Metrics:**

* **Mean Squared Error (MSE):** 610,861,376.00
* **Root Mean Squared Error (RMSE):** 24,715.61
* **R-squared (R2):** 0.9204

The high R-squared value indicates that the model effectively captures a significant portion (over 92%) of the variance in house prices.

**Kaggle Competition Score:** 0.13469
*(Note: A lower score indicates better performance in this competition, which typically uses RMSE.)*

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Author

Abhinav Marlingaplar
