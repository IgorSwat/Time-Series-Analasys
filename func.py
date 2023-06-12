import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from math import inf

def df_from_years(df, beg_year, end_year):
    left_key = '01-01-' + beg_year
    right_key = '31-12-' + end_year
    return df[left_key:right_key]


def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16, 5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()


def decomposition(df, decomp_model):
    if decomp_model == 'additive':
        result = seasonal_decompose(df['value'], model='additive', extrapolate_trend='freq')
    elif decomp_model == 'multiplicative':
        result = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq')
    else:
        print("Invalid decomposition model name:", decomp_model)
        return
    plt.rcParams.update({'figure.figsize': (10, 10)})
    result.plot().suptitle(decomp_model + ' decomposition', fontsize=22)
    plt.show()


def deseasonalisation(df, title=""):
    result = seasonal_decompose(df['value'], model='additive', extrapolate_trend='freq')
    deseasonalized = df.value.values - result.seasonal
    plt.plot(deseasonalized)
    plt.title(title, fontsize=16)
    plt.show()


def detrend(df, title=""):
    result = seasonal_decompose(df['value'], model='additive', extrapolate_trend='freq')
    detrended = df.value.values - result.trend
    plt.plot(detrended)
    plt.title(title, fontsize=16)
    plt.show()


def plot_differences(df, n):
    plt.rcParams.update({'figure.figsize': (12, 6), 'figure.dpi': 120})

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    lags = len(df) - 1
    axes[0, 0].plot(df.index, df['value'])
    axes[0, 0].set_title('Original Series')
    plot_acf(df['value'].values, ax=axes[0, 1], lags=lags)
    diff = df['value']
    for i in range(n):
        diff = diff.diff()
    axes[1, 0].plot(df.index, diff)
    axes[1, 0].set_title(str(n) + 'th Order Differencing')
    plot_acf(diff.dropna().values, ax=axes[1, 1], lags=lags - n, zero=False)
    start_date = df.index[1]  # Exclude the first element due to differencing
    end_date = df.index[-1]
    axes[0, 0].set_xlim(start_date, end_date)
    axes[0, 1].set_xlim(0, lags)
    axes[1, 0].set_xlim(start_date, end_date)
    axes[1, 1].set_xlim(0, lags - n)
    plt.tight_layout()
    plt.show()


def plot_AR(df, n):
    plt.rcParams.update({'figure.figsize': (12, 6), 'figure.dpi': 120})
    fig, axes = plt.subplots(1, 2)
    diff = df['value']
    for i in range(n):
        diff = diff.diff()
    axes[0].plot(diff)
    axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0, 5))
    plot_pacf(diff.dropna(), ax=axes[1])
    plt.show()


def plot_MA(df, n):
    plt.rcParams.update({'figure.figsize': (12, 6), 'figure.dpi': 120})
    fig, axes = plt.subplots(1, 2)
    diff = df['value']
    for _ in range(n):
        diff = diff.diff()
    axes[0].plot(diff)
    axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0, 1.2))
    plot_acf(diff.dropna(), ax=axes[1], lags=len(df)- n - 1)
    plt.show()


def find_params(df, break_point):
    train_data = df.loc[df.index < break_point]
    test_data = df.loc[df.index >= break_point]
    periods = len(test_data)
    lse_min = inf
    AR, MA = 1, 1
    for i in range(1, 5):
        for j in range(i, 30):
            model = ARIMA(train_data['value'], order=(1, i, j))
            model_fit = model.fit()
            forc = model_fit.get_forecast(steps=periods)
            forecasted_values = forc.predicted_mean
            stderr = forc.se_mean
            forecast_index = pd.date_range(start=test_data.index[0], periods=periods,
                                           freq=test_data.index.inferred_freq)
            forecast_df = pd.DataFrame({'Forecast': forecasted_values, 'StdErr': stderr}, index=forecast_index)
            arr1 = test_data['value'].values
            arr2 = forecast_df['Forecast'].values
            lse = np.linalg.norm(arr1 - arr2)
            if lse < lse_min:
                lse_min = lse
                AR = i
                MA = j
    return AR, MA


def forecast1(df, break_point):
    plt.rcParams.update({'figure.figsize': (12, 6), 'figure.dpi': 120})
    train_data = df.loc[df.index < break_point]
    test_data = df.loc[df.index >= break_point]
    model = ARIMA(train_data['value'], order=(1, 3, 12))
    model_fit = model.fit()
    print(model_fit.summary())
    periods = len(test_data)
    forc = model_fit.get_forecast(steps=periods)
    forecasted_values = forc.predicted_mean
    stderr = forc.se_mean
    conf_int = forc.conf_int()
    forecast_index = pd.date_range(start=test_data.index[0], periods=periods, freq=test_data.index.inferred_freq)
    forecast_df = pd.DataFrame({'Forecast': forecasted_values, 'StdErr': stderr}, index=forecast_index)
    plt.plot(test_data.index, test_data['value'], label='Actual Values')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecasted Values')
    plt.fill_between(forecast_df.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='gray', alpha=0.3,
                     label='Confidence Intervals')
    plt.legend()
    plt.show()


def forecast2(df, year):
    plt.rcParams.update({'figure.figsize': (12, 6), 'figure.dpi': 120})
    model = ARIMA(df['value'], order=(1, 3, 12))
    model_fit = model.fit()
    index_year = pd.to_datetime(df.index[-1]).year
    periods = int(year) - index_year
    forc = model_fit.get_forecast(steps=periods)
    forecasted_values = forc.predicted_mean
    stderr = forc.se_mean
    forecast_index = pd.date_range(start=df.index[-1], periods=periods, freq=df.index.inferred_freq)
    forecast_df = pd.DataFrame({'Forecast': forecasted_values, 'StdErr': stderr}, index=forecast_index)
    plt.plot(df.index, df['value'], label='Actual Values')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecasted Values')
    plt.legend()
    plt.show()