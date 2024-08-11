
# Car Price Prediction Model

This project involves developing a machine learning model to predict car prices based on various features, including car make, model, year of manufacture, kilometers driven, and fuel type. The project includes data cleaning, exploratory data analysis (EDA), feature engineering, model training, and evaluation.
## Features

- Data Cleaning: Handling missing values, converting data types, and normalizing columns.
- Exploratory Data Analysis (EDA): Visualization of price distributions and feature relationships.
- Feature Engineering: One-Hot Encoding for categorical features and selection of relevant features.
- Model Building: Linear regression model training and performance evaluation.
- Prediction: Function for predicting car prices for new data.
- Model Deployment: Saving the trained model for future use.
## Getting Started

1. Python 3.x: Ensure Python 3.x is installed.
2. Ensure you have the following Python libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pickle

You can install the necessary libraries using pip:

```bash
  pip install pandas numpy matplotlib seaborn scikit-learn
```
## Usage

1. Prepare Data: Place your dataset (carprizedataset.csv) in the project directory.
2. Run the Analysis: Execute the Jupyter Notebook car_price_prediction.ipynb to perform data cleaning, EDA, model training, and evaluation.
3. Predict Prices: Use the provided function to predict car prices based on new data:
```python
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Predict car price
new_car = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                       data=[['Maruti Suzuki Swift', 'Maruti', 2019, 100, 'Petrol']])
price_prediction = model.predict(new_car)
print(f"Predicted Price: {price_prediction[0]}")

```
4. Evaluate Model: Use the R² score to evaluate model performance.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug fixes, feel free to fork the repository and submit a pull request. Please follow the project's code of conduct.

Steps to Contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/YourFeature).
3. Commit your changes (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/YourFeature).
5. Open a pull request.

