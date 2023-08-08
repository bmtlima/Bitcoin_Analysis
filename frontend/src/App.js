import React, { useEffect, useState } from 'react';
import './App.css';
import axios from 'axios';
import Plot from 'react-plotly.js';

function App() {
  const [data, setData] = useState({});

  useEffect(() => {
    // Fetch data from the Flask backend
    axios.get('/data')
      .then(response => {
        setData(response.data);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
      });
  }, []);

  return (
    <div className="App">
      <h1>Cryptocurrency Data Visualization</h1>
      <h2>RMSE: {data.rmse}</h2>
      <h2>MAE: {data.mae}</h2>
      <h2>Percentage Error: {data.percentage_error}%</h2>
      <Plot
        data={[
          {
            x: data.macd_line,
            y: data.signal_line,
            type: 'scatter',
            mode: 'lines',
            name: 'MACD Line',
          },
          {
            x: data.macd_line,
            y: data.macd_histogram,
            type: 'bar',
            name: 'MACD Histogram',
          },
        ]}
        layout={{ title: 'MACD Indicator' }}
      />
    </div>
  );
}

export default App;
