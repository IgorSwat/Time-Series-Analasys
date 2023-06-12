import pandas as pd
from func import plot_df
from func import df_from_years
from func import detrend
from func import decomposition
from func import deseasonalisation
from func import plot_differences
from func import plot_AR
from func import plot_MA
from func import forecast1
from func import find_params
from func import forecast2


def main():
    df = pd.read_csv('temperatures.csv', parse_dates=['date'], date_format="%Y")
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    print("Data succesfuly loaded: Average world's April temperature anomalies")
    print("Select analysed years (1850-2023)")
    beg_year = input("Select begining year: ")
    end_year = input("Select end year: ")
    df_selected = df_from_years(df, beg_year, end_year)
    print("Available time series analyse operations: plot, decomposition, deseasonalize, detrend, differences, AR, MA, forecast")
    operation = input("Select operation: ")
    if operation == "plot":
        plot_df(df_selected, df_selected.index, df_selected.values, "Average world's April temperature anomalies in years " +
                beg_year + " - " + end_year, xlabel="Year", ylabel="Value [C]")
    elif operation == "decomposition":
        model = input("Select decomposition model (additive, multiplicative): ")
        decomposition(df_selected, model)
    elif operation == "deseasonalize":
        deseasonalisation(df_selected, "Average world's April temperature anomalies in years " + beg_year + " - " + end_year + " deseasonalized")
    elif operation == "detrend":
        detrend(df_selected, "Average world's April temperature anomalies " + beg_year + " - " + end_year + " detrended")
    elif operation == "differences":
        deg = input("Select degree of differencing: ")
        plot_differences(df_selected, int(deg))
    elif operation == "AR":
        deg = input("Select degree of differencing: ")
        plot_AR(df_selected, int(deg))
    elif operation == "MA":
        deg = input("Select degree of differencing: ")
        plot_MA(df_selected, int(deg))
    elif operation == "forecast":
        choice = input("Select forecast options (comparision, future): ")
        if choice == "comparision":
            break_point = input("Select training data year ending: ")
            forecast1(df_selected, break_point)
        elif choice == "future":
            year = input("Select ending year for forecast: ")
            forecast2(df, year)
        else:
            print("Unresolved command:", choice)
            exit(1)
    else:
        print("Unresolved command:", operation)
        exit(1)


def preprocess():
    df = pd.read_csv('temperatures.csv', parse_dates=['date'], date_format="%Y")
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    AR, MA = find_params(df, "2000")
    print(AR, MA)


if __name__ == "__main__":
    main()