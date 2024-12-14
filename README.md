
# Stock Market Prediction App using LSTM  

This repository contains a **Streamlit** web application that uses a **Long Short-Term Memory (LSTM)** neural network to predict stock market prices. The app allows users to upload a dataset, visualize trends, and forecast future stock prices interactively.  

---

## Features  

- **Upload CSV Dataset**: Easily upload historical stock market data in CSV format.  
- **Data Visualization**: Visualize historical prices and model predictions with interactive Plotly graphs.  
- **LSTM Model for Predictions**: Use an LSTM neural network to predict future stock prices.  
- **Customizable Future Predictions**: Forecast stock prices for a user-defined number of days.  
- **Downloadable Results**: Export future predictions as a CSV file.  

---

## Getting Started  

Follow these instructions to set up and run the project locally.  

### Prerequisites  

Ensure the following are installed:  

- Python 3.8 or later  
- pip (Python package manager)  

### Installation  

1. Clone this repository:  
   ```bash  
   git clone https://github.com/your-username/stock-prediction-app.git  
   cd stock-prediction-app  
   ```  

2. Install the required dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. Run the application:  
   ```bash  
   streamlit run app.py  
   ```  

4. Open your browser at `http://localhost:8501` to interact with the app.  

---

## Usage  

1. **Upload a Dataset**:  
   Upload a CSV file containing stock data with at least one column for closing prices.  

2. **Select Closing Price Column**:  
   Choose the column representing the closing prices from a dropdown menu.  

3. **View Predictions**:  
   The app splits the data into training and testing sets, trains an LSTM model, and displays predictions on an interactive graph.  

4. **Forecast Future Prices**:  
   Enter the number of future days to predict. The app will display the forecasted prices and allow you to download them as a CSV file.  

---

## Dataset Requirements  

- The uploaded file must be in CSV format.  
- The dataset should include a column with closing prices.  

Example dataset format:  

| Date       | Open   | High   | Low    | Close  | Volume   |  
|------------|--------|--------|--------|--------|----------|  
| 2023-01-01 | 100.0  | 110.0  | 99.0   | 105.0  | 1000000  |  
| 2023-01-02 | 106.0  | 112.0  | 101.0  | 110.0  | 1200000  |  

---

## Technologies Used  

- **Python**: Core programming language.  
- **Streamlit**: For creating the web application.  
- **Keras**: For building the LSTM neural network.  
- **TensorFlow**: Backend for Keras.  
- **Pandas**: For data handling and manipulation.  
- **Numpy**: For numerical computations.  
- **Plotly**: For interactive visualizations.  

---

## Deployment on Streamlit Cloud  

To deploy this app on **Streamlit Cloud**:  

1. Push this project to a GitHub repository.  
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and log in.  
3. Create a new app, link your repository, and deploy.  

Ensure the `requirements.txt` file is included in your repository for dependencies.  

---

## Future Enhancements  

- Add support for multiple machine learning models.  
- Improve handling of missing or noisy data.  
- Provide additional performance metrics such as RMSE or MAE.  
- Enable predictions for multiple stocks in a single session.  

---

## License  

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  

---

## Acknowledgments  

- [Streamlit](https://streamlit.io/) for the intuitive web framework.  
- [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) for deep learning support.  
- [Plotly](https://plotly.com/) for powerful and interactive visualizations.  

---

## Screenshots  

![image](https://github.com/user-attachments/assets/8ad9dbed-533d-4124-bd57-d90be0919309)
 

 

---

## Contributions  

Contributions are welcome! Feel free to submit a pull request or open an issue for any bugs or feature requests.  
