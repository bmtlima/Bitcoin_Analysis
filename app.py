from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)

# Load and preprocess the cryptocurrency data
df = pd.read_csv('raw.githubusercontent.com_coinmetrics_data_master_csv_btc.csv', low_memory=False)
df = df.drop(df.index[-1])
null_proportions = df.isnull().mean()
columns_with_nan = df.columns[df.isna().any()].tolist()
df['time'] = pd.to_datetime(df['time'])
start_date = pd.to_datetime("2011-08-29")
filtered_df = df.loc[df['time'] >= start_date].copy()
clean_df = filtered_df.drop(["CapMrktEstUSD", "ReferenceRateETH", "principal_market_price_usd", "principal_market_usd"], axis=1)
constant_columns = clean_df.columns[clean_df.nunique() == 1].tolist()
clean_df = clean_df.drop(constant_columns, axis=1)
df_rmse = clean_df.copy()
df_rmse['time'] = (df_rmse['time'] - df_rmse['time'].min()).dt.days

# Initialize the Rolling Window Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

# Perform Rolling Window Cross-Validation and calculate RMSE, MAE, and Percentage Error
actual_values = []
predicted_values = []


for train_index, test_index in tscv.split(df_rmse):
    train_data = df_rmse.iloc[train_index]
    test_data = df_rmse.iloc[test_index]
    model = LinearRegression()
    model.fit(train_data[['time']], train_data['PriceUSD'])
    predictions = model.predict(test_data[['time']])
    actual_values.extend(test_data['PriceUSD'])
    predicted_values.extend(predictions)

actual_values = np.array(actual_values)
predicted_values = np.array(predicted_values)

rmse = mean_squared_error(actual_values, predicted_values, squared=False)
mae = mean_absolute_error(actual_values, predicted_values)
percentage_error = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

# Calculate MACD and trading strategy metrics
test_df = clean_df.copy()
ema_12 = test_df['PriceUSD'].ewm(span=12, adjust=False).mean()
ema_26 = test_df['PriceUSD'].ewm(span=26, adjust=False).mean()
macd_line = ema_12 - ema_26
signal_line = macd_line.ewm(span=9, adjust=False).mean()
macd_histogram = macd_line - signal_line

buy_threshold = 0
sell_threshold = 0
position = None
positions = []
pl = []

for i in range(1, len(test_df)):
    if macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i - 1] < signal_line.iloc[i - 1] and (position is None or position == 'SELL'):
        position = 'BUY'
        positions.append((test_df['time'].iloc[i], test_df['PriceUSD'].iloc[i]))
    if macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i - 1] > signal_line.iloc[i - 1] and (position is None or position == 'BUY'):
        position = 'SELL'
        positions.append((test_df['time'].iloc[i], test_df['PriceUSD'].iloc[i]))
    if position is not None:
        if position == 'BUY':
            pl.append(test_df['PriceUSD'].iloc[i] - positions[-1][1])
        else:
            pl.append(positions[-1][1] - test_df['PriceUSD'].iloc[i])

# Organize data for sending to frontend
data = {
    'rmse': rmse,
    'mae': mae,
    'percentage_error': percentage_error,
    'macd_line': macd_line.tolist(),
    'signal_line': signal_line.tolist(),
    'macd_histogram': macd_histogram.tolist(),
    'pl': pl,
}

# Define a route to serve JSON data to the frontend
@app.route('/data')
def get_data():
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
