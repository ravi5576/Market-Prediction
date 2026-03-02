# 📈 Stock Market Prediction Web App

A Flask-based web application that predicts future stock prices using a trained LSTM model. It enables users to explore and visualize stock market trends with real-time data fetched from Yahoo Finance.

## 🚀 Features

- User Authentication (Login/Register with session management)
- Admin Dashboard for monitoring feedback
- Fetch historical stock data from Yahoo Finance
- Predict next 10 days of stock prices using a deep learning model
- Graphs:
  - Actual vs Predicted Prices
  - Next Day Prediction
  - 10-Day Price Forecast
  - Closing Price History
  - Moving Averages (100 & 200 days)
- Mutual fund and stock type identification
- Feedback submission and storage in SQLite

## 🧠 Technologies Used

- Flask (Python backend framework)
- LSTM Model (TensorFlow/Keras)
- SQLite (Feedback database)
- Yahoo Finance API (via `yfinance`)
- Matplotlib (Data visualization)
- Pandas, NumPy, Scikit-learn

## 🔐 User Roles

- **Admin**: Access feedback and user list
- **User**: Access prediction tools and tutorials

## 📂 Project Structure
```
/templates
index.html
login.html
results.html
tutorial.html
admin_dashboard.html
/static
/css
/js
model/
stock_future_prediction_saved.keras
app.py
feedback.db
```

## 🛠 How to Run

1. Clone the repository
2. Install required packages:  
   `pip install -r requirements.txt`
3. Run the Flask app:  
   `python app.py`
4. Open browser and go to:  
   `http://localhost:5000/`

## 🧪 Demo Users

- **Admin**  
  Username: `admin`  
  Password: `password123`

- **User**  
  Username: `user1`  
  Password: `userpass`

## 📬 Feedback

Feedback from users is stored in a local SQLite database and viewable in the admin dashboard.

---

