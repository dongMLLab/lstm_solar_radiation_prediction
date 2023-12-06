# Solar Radiation Prediction

## DataSet

- Climate ASOS data collected by South Korean Meteorological Administration(KMA)
- Hourly Data Period: 2017-01-01 00:00:00 ~ 2023-10-18 23:00:00

## Methods

- Long Short-Term memory(LSTM)
- time_steps: 48

### Feature Selection

- Top Attributes selected By Correlations(Pearson)
- All Attributes

### Preprocessing

- 1. Without Scaling
- 2. MinMaxScaling: range 0 - 1

## Conclusion

- Better predicting performance results with all features following evaluation stats(MSE, RMSE)
    - With all attributes: 0.012, 0.000
    - With correlated attributes: 0.016, 0.127

- Better results using correlated features with relatively robustic results.
- All Attributes results following raw graph shapes
