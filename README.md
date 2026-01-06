# 🛒 Walmart Sales Predictor (Linear Regression)

A Streamlit application that predicts weekly sales for Walmart stores using a Linear Regression model. This tool allows users to input various economic and environmental parameters to estimate sales performance.

## 📊 Features

- **Sales Prediction**: Real-time weekly sales prediction based on inputs like Store ID, Date, Fuel Price, CPI, and Unemployment rate.
- **Historical Trends**: Interactive visualization of historical sales data for selected stores.
- **Machine Learning**: Utilizes Scikit-Learn's Linear Regression for on-the-fly model training and prediction.
- **User-Friendly Interface**: Clean and intuitive UI built with Streamlit.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Vibin-007/linear_regression.git
   cd linear_regression
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

- `app.py`: Main application file containing the Streamlit interface and logic.
- `Walmart_Sales.csv`: The dataset used for training the model.
- `walmart_sales_analysis.ipynb`: Jupyter notebook for exploratory data analysis and model experimentation.
- `requirements.txt`: List of Python dependencies.

## 📈 Model Information

The model uses **Linear Regression** to predict 'Weekly_Sales' based on:
- **Store Number**
- **Holiday Flag**
- **Temperature**
- **Fuel Price**
- **CPI** (Consumer Price Index)
- **Unemployment Rate**