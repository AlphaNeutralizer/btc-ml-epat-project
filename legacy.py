import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, ADXIndicator
import psutil

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm.contrib.concurrent import process_map


print(vbt.__version__)

DEFAULT_DATA_ROOT = Path("data/monthly/BTC-USD")
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def timeframe_to_freq(timeframe: str) -> str:
    timeframe = timeframe.lower().strip()
    units = {"m": "min", "h": "h", "d": "d"}

    if len(timeframe) < 2 or timeframe[-1] not in units:
        raise ValueError(
            "Timeframe must use a supported suffix, such as 1m, 15m, 1h, or 1d."
        )

    value = timeframe[:-1]
    if not value.isdigit() or int(value) <= 0:
        raise ValueError("Timeframe must start with a positive integer.")

    return f"{int(value)}{units[timeframe[-1]]}"


def normalize_freq(freq: str) -> str:
    try:
        return timeframe_to_freq(freq)
    except ValueError:
        return freq


def load_timeframe_data(timeframe: str, data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    timeframe = timeframe.lower().strip()
    data_dir = data_root / timeframe

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Could not find data directory: {data_dir}")

    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in data directory: {data_dir}")

    dataframes = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df.columns = [column.strip().lower() for column in df.columns]

        if "timestamp" not in df.columns:
            df = df.rename(columns={df.columns[0]: "timestamp"})

        missing_columns = {"timestamp", *OHLCV_COLUMNS} - set(df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        for column in OHLCV_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        dataframes.append(df[["timestamp", *OHLCV_COLUMNS]])

    dataframe = pd.concat(dataframes, ignore_index=True)
    dataframe = dataframe.dropna(subset=["timestamp", "open", "high", "low", "close"])
    dataframe = dataframe.sort_values("timestamp")
    dataframe = dataframe.drop_duplicates(subset="timestamp", keep="last")
    dataframe = dataframe.set_index("timestamp")

    return dataframe


def fit_model_backtest(kv_pairs: dict) -> tuple[int, dict[str, object]]:
    # getting the parameters from the input dictionary
    dataframe = kv_pairs["dataframe"]
    num_estimators = kv_pairs["num_estimators"]
    freq = kv_pairs["freq"]
    resample = kv_pairs["resample"]

    # creating a copy of the dataframe which will have the indicators applied to it
    df = dataframe[["open", "high", "low", "close"]].copy(deep=True)

    # Resampling is optional because the input CSVs can already use the target timeframe.
    if resample:
        ohlc_dict = {"open": "first", "high": "max", "low": "min", "close": "last"}
        df = df.resample(resample).apply(ohlc_dict)
        df = df.dropna(subset=["open", "high", "low", "close"])

    # indicators applied to the data, a standard 14 period is used for all the indicators, keeping the strategy simple
    # I have used the Random forest algorithm which is basically a collection of decision trees, hence it doesn't require
    # scaling of data. Additionally all the indicators I have used range from 0-100 or -1 to 1 as I take ratios for those that
    # don't have upper and lower bounds, hence the rules to tend to persist for longer periods.
    timeperiod = 14

    df["pct_change"] = df["close"].pct_change()
    df["pct_change_15"] = df["close"].pct_change(15)
    df["rsi"] = RSIIndicator(df["close"], window=timeperiod).rsi()
    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], window=timeperiod).adx()
    df["sma"] = SMAIndicator(df["close"], window=timeperiod).sma_indicator()
    df["sma/close"] = df["sma"] / df["close"]
    df["corr"] = df["close"].rolling(timeperiod).corr(df["sma"])
    df["volatility"] = df["pct_change"].rolling(timeperiod).std() * 100
    df["volatility_2"] = df["pct_change_15"].rolling(timeperiod).std() * 100
    df["future_pct_change"] = df["pct_change"].shift(-1)
    # df["future_signal"] = np.where(df["future_pct_change"] > 0, 1, 0)
    df["future_signal"] = np.where(
        df["future_pct_change"] > 0, 1, np.where(df["future_pct_change"] < 0, -1, 0)
    )
    df = df.replace([np.inf, -np.inf], np.nan)

    inputs = [
        "pct_change",
        "pct_change_15",
        "rsi",
        "adx",
        "sma/close",
        "corr",
        "volatility",
        "volatility_2",
    ]

    output = "future_signal"

    # using 75% 25% train-test split
    model_window = (3 * df.shape[0]) // 4
    train_data = df.iloc[:model_window].copy(deep=True)
    test_data = df.iloc[model_window:].copy(deep=True)

    # using a Random Forest classifier model to make discrete buy, sell, and hold signals
    model = RandomForestClassifier(
        n_estimators=num_estimators, criterion="gini", random_state=0
    )
    model.fit(train_data[inputs], train_data[output])

    # creating a column for the predictions
    train_data["forecast"] = model.predict(train_data[inputs])
    test_data["forecast"] = model.predict(test_data[inputs])

    # signal is used to buy or sell, essentially the forecast shifted by one to account for the delay in placing the
    # trades
    train_data["signal"] = train_data["forecast"].shift(1)
    test_data["signal"] = test_data["forecast"].shift(1)

    # creating the entries and exits for both the long and short trades.
    test_long_entries = test_data["signal"] == 1
    test_short_entries = test_data["signal"] == -1

    test_long_exits = test_data["signal"].shift(-1) != 1
    test_short_exits = test_data["signal"].shift(-1) != -1

    # using Vectorbt to run a vectorized backtest on the training data
    test_pf = vbt.Portfolio.from_signals(
        test_data["close"],
        entries=test_long_entries,
        exits=test_long_exits,
        short_entries=test_short_entries,
        short_exits=test_short_exits,
        freq=freq,
        size_granularity=1e-8,
    )

    stats = test_pf.stats()

    # tabulating accuracy of the model and other important backtest parameters.
    accuracy = (
        accuracy_score(test_data[output], test_data["forecast"], normalize=True) * 100
    )

    return (
        num_estimators,
        {
            "Sharpe Ratio": round(stats["Sharpe Ratio"], 2),
            "Excess Return [%]": round(
                stats["Total Return [%]"] - stats["Benchmark Return [%]"], 2
            ),
            "Win Rate [%]": round(stats["Win Rate [%]"], 2),
            "Accuracy [%]": round(accuracy, 2),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BTC-USD random forest backtest.")
    parser.add_argument(
        "--timeframe",
        default="1d",
        help="Timeframe folder to load under the data root, such as 1m, 15m, 1h, or 1d.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory that contains timeframe folders.",
    )
    parser.add_argument(
        "--resample",
        default=None,
        help="Optional frequency to resample input data before backtesting, such as 15m, 15min, or 1h.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path for KPI results. Defaults to KPI_<timeframe>.csv.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=psutil.cpu_count(logical=True),
        help="Maximum number of worker processes to use for backtests.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    timeframe = args.timeframe.lower().strip()
    resample = normalize_freq(args.resample) if args.resample else None
    freq = resample or timeframe_to_freq(timeframe)

    # loading the raw data
    dataframe = load_timeframe_data(timeframe, args.data_root)
    output_path = args.output or Path(f"KPI_{timeframe}.csv")

    # creating the various key value pairs that will be passed for backtesting
    # the hyperparameter being optimized for is the number of esitmators, i.e the number of decision trees in the model
    kv_pairs_list = [
        {
            "dataframe": dataframe,
            "num_estimators": num_estimators,
            "freq": freq,
            "resample": resample,
        }
        for num_estimators in range(5, 26, 1)
    ]

    # using a process pool to run the backtests so that multiple can run in parallel
    # if you are facing CPU temp or RAM usage constraints, consider seting logical=False for the cpu_count
    # or even reduce the number of workers in the max_workers param, this will reduce system resource utilization
    # but your backtests will take longer to complete.
    # for reference, this backtest took me ~6 minutes.
    backtest_results = dict(
        process_map(
            fit_model_backtest,
            kv_pairs_list,
            max_workers=args.max_workers,
            desc="Backtests",
        )
    )

    # saving the backtest results
    kpis_df = pd.DataFrame(backtest_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kpis_df.to_csv(output_path)


if __name__ == "__main__":
    main()