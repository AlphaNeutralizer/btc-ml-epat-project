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
DEFAULT_BEST_METRIC = "Sharpe Ratio"


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
    freq = freq.lower().strip()
    if len(freq) >= 2 and freq[-1] in {"m", "h", "d"} and freq[:-1].isdigit():
        return timeframe_to_freq(freq)
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


def validate_non_overlapping_signals(signals: dict[str, pd.Series]) -> None:
    signal_items = [
        (name, series.fillna(False).astype(bool)) for name, series in signals.items()
    ]

    for left_index, (left_name, left_series) in enumerate(signal_items):
        for right_name, right_series in signal_items[left_index + 1 :]:
            overlap = left_series & right_series
            if not overlap.any():
                continue

            overlap_index = overlap[overlap].index
            examples = ", ".join(str(index_value) for index_value in overlap_index[:5])
            if len(overlap_index) > 5:
                examples = f"{examples}, ..."
            raise ValueError(
                f"Signal overlap detected between '{left_name}' and '{right_name}' "
                f"at: {examples}"
            )


def build_trade_events(signal: pd.Series, allow_short: bool) -> dict[str, pd.Series]:
    events = {
        "long_entries": pd.Series(False, index=signal.index),
        "long_exits": pd.Series(False, index=signal.index),
        "short_entries": pd.Series(False, index=signal.index),
        "short_exits": pd.Series(False, index=signal.index),
    }
    position = 0

    for timestamp, signal_value in signal.items():
        target = 0
        if signal_value == 1:
            target = 1
        elif allow_short and signal_value == -1:
            target = -1

        if position == 0:
            if target == 1:
                events["long_entries"].loc[timestamp] = True
                position = 1
            elif target == -1:
                events["short_entries"].loc[timestamp] = True
                position = -1
        elif position == 1 and target != 1:
            events["long_exits"].loc[timestamp] = True
            position = 0
        elif position == -1 and target != -1:
            events["short_exits"].loc[timestamp] = True
            position = 0

    validate_non_overlapping_signals(events)
    return events


def run_model_backtest(
    dataframe: pd.DataFrame,
    num_estimators: int,
    freq: str,
    resample: str | None,
    allow_short: bool,
) -> dict[str, object]:
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

    # creating entry/exit events from signal state transitions.
    trade_events = build_trade_events(test_data["signal"], allow_short)
    test_long_entries = trade_events["long_entries"]
    test_long_exits = trade_events["long_exits"]
    test_short_entries = trade_events["short_entries"]
    test_short_exits = trade_events["short_exits"]
    test_buy_signals = test_long_entries
    test_sell_signals = test_long_exits

    validate_non_overlapping_signals(
        {
            "buy_signals": test_buy_signals,
            "sell_signals": test_sell_signals,
        }
    )

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
    stats = stats.copy()
    stats["Excess Return [%]"] = stats["Total Return [%]"] - stats["Benchmark Return [%]"]
    stats["Accuracy [%]"] = accuracy

    summary_stats = {
        "Sharpe Ratio": round(stats["Sharpe Ratio"], 2),
        "Excess Return [%]": round(stats["Excess Return [%]"], 2),
        "Win Rate [%]": round(stats["Win Rate [%]"], 2),
        "Accuracy [%]": round(accuracy, 2),
    }

    return {
        "portfolio": test_pf,
        "stats": stats,
        "summary_stats": summary_stats,
        "train_data": train_data,
        "test_data": test_data,
        "long_entries": test_long_entries,
        "short_entries": test_short_entries,
        "long_exits": test_long_exits,
        "short_exits": test_short_exits,
        "buy_signals": test_buy_signals,
        "sell_signals": test_sell_signals,
        "allow_short": allow_short,
    }


def fit_model_backtest(kv_pairs: dict) -> tuple[int, dict[str, object]]:
    # getting the parameters from the input dictionary
    dataframe = kv_pairs["dataframe"]
    num_estimators = kv_pairs["num_estimators"]
    freq = kv_pairs["freq"]
    resample = kv_pairs["resample"]
    allow_short = kv_pairs["allow_short"]

    result = run_model_backtest(dataframe, num_estimators, freq, resample, allow_short)

    return (
        num_estimators,
        result["summary_stats"],
    )


def select_plot_estimator(
    kpis_by_estimator: pd.DataFrame,
    plot_estimator: int | None,
    best_metric: str,
) -> int:
    if plot_estimator is not None:
        if plot_estimator not in kpis_by_estimator.index:
            raise ValueError(
                f"Estimator {plot_estimator} was not in the backtest sweep results."
            )
        return plot_estimator

    if best_metric not in kpis_by_estimator.columns:
        available_metrics = ", ".join(kpis_by_estimator.columns)
        raise ValueError(
            f"Best metric '{best_metric}' was not found. Available metrics: {available_metrics}"
        )

    metric_values = pd.to_numeric(kpis_by_estimator[best_metric], errors="coerce")
    if metric_values.dropna().empty:
        raise ValueError(f"Best metric '{best_metric}' has no numeric values.")

    return int(metric_values.idxmax())


def format_plotly_metric(value: object) -> str:
    if value is None:
        return ""

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.Timedelta):
        return str(value)

    if isinstance(value, np.integer):
        return f"{int(value):,}"

    if isinstance(value, int):
        return f"{value:,}"

    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return ""
        return f"{float(value):,.4f}".rstrip("0").rstrip(".")

    return str(value)


def default_plot_output_path(timeframe: str, resample: str | None) -> Path:
    if not resample:
        return Path(f"backtest_report_{timeframe}.html")

    safe_resample = resample.replace("/", "-").replace("\\", "-")
    return Path(f"backtest_report_{timeframe}_resampled_{safe_resample}.html")


def write_plotly_report(
    result: dict[str, object],
    output_path: Path,
    title: str,
    num_estimators: int,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise ImportError(
            "Plotly is required to create the backtest report. Install it with: pip install plotly"
        ) from error

    portfolio = result["portfolio"]
    stats = result["stats"]
    test_data = result["test_data"]
    buy_signals = result["buy_signals"]
    sell_signals = result["sell_signals"]

    equity = portfolio.value()
    if isinstance(equity, pd.DataFrame):
        equity = equity.iloc[:, 0]

    btc_price = test_data["close"]
    benchmark = test_data["close"] / test_data["close"].iloc[0] * equity.iloc[0]
    cumulative_return = (equity / equity.iloc[0] - 1) * 100
    positive_return = cumulative_return.where(cumulative_return >= 0)
    negative_return = cumulative_return.where(cumulative_return < 0)
    drawdown = (equity / equity.cummax() - 1) * 100
    buy_signal_equity = equity.loc[buy_signals[buy_signals].index]
    sell_signal_equity = equity.loc[sell_signals[sell_signals].index]

    stats_df = stats.reset_index()
    stats_df.columns = ["Metric", "Value"]
    stats_df["Value"] = stats_df["Value"].map(format_plotly_metric)
    figure_height = max(1240, 760 + len(stats_df) * 24)

    sharpe = format_plotly_metric(stats.get("Sharpe Ratio"))
    total_return = format_plotly_metric(stats.get("Total Return [%]"))
    benchmark_return = format_plotly_metric(stats.get("Benchmark Return [%]"))
    max_drawdown = format_plotly_metric(stats.get("Max Drawdown [%]"))

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "table"}],
        ],
        row_heights=[0.29, 0.22, 0.2, 0.29],
        vertical_spacing=0.06,
        subplot_titles=(
            "Equity Curve",
            "BTC Price",
            "Return / Drawdown [%]",
            "Vectorbt Stats",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="Strategy Equity",
            line={"color": "#2563eb", "width": 2},
        ),
        row=1,
        col=1,
    )
    if not buy_signal_equity.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signal_equity.index,
                y=buy_signal_equity.values,
                mode="markers",
                name="Buy Signal",
                marker={
                    "color": "#16a34a",
                    "symbol": "triangle-up",
                    "size": 10,
                    "line": {"color": "white", "width": 1},
                },
            ),
            row=1,
            col=1,
        )
    if not sell_signal_equity.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signal_equity.index,
                y=sell_signal_equity.values,
                mode="markers",
                name="Sell Signal",
                marker={
                    "color": "#dc2626",
                    "symbol": "triangle-down",
                    "size": 10,
                    "line": {"color": "white", "width": 1},
                },
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            mode="lines",
            name="Buy and Hold",
            line={"color": "#64748b", "dash": "dash", "width": 1.5},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=btc_price.index,
            y=btc_price.values,
            mode="lines",
            name="BTC Close",
            line={"color": "#0f766e", "width": 1.6},
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=positive_return.index,
            y=positive_return.values,
            mode="lines",
            fill="tozeroy",
            connectgaps=False,
            name="Positive Return [%]",
            line={"color": "#16a34a", "width": 1.5},
            fillcolor="rgba(22, 163, 74, 0.25)",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=negative_return.index,
            y=negative_return.values,
            mode="lines",
            fill="tozeroy",
            connectgaps=False,
            name="Negative Return [%]",
            line={"color": "#dc2626", "width": 1.5},
            fillcolor="rgba(220, 38, 38, 0.25)",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            name="Drawdown [%]",
            line={"color": "#991b1b", "width": 1.4, "dash": "dot"},
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Table(
            header={
                "values": ["Metric", "Value"],
                "fill_color": "#111827",
                "font": {"color": "white", "size": 12},
                "align": "left",
            },
            cells={
                "values": [stats_df["Metric"].tolist(), stats_df["Value"].tolist()],
                "fill_color": "#f8fafc",
                "font": {"color": "#111827", "size": 11},
                "align": "left",
                "height": 24,
            },
        ),
        row=4,
        col=1,
    )

    fig.update_layout(
        title=(
            f"{title}<br>"
            f"<sup>Estimator: {num_estimators} | Sharpe: {sharpe} | "
            f"Return: {total_return}% | Benchmark: {benchmark_return}% | "
            f"Max Drawdown: {max_drawdown}%</sup>"
        ),
        template="plotly_white",
        height=figure_height,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 70, "r": 40, "t": 110, "b": 50},
    )
    fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
    fig.update_yaxes(title_text="BTC Close", row=2, col=1)
    fig.update_yaxes(title_text="Percent", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True)


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
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Output HTML path for the Plotly report. Defaults to backtest_report_<timeframe>.html.",
    )
    parser.add_argument(
        "--plot-estimator",
        type=int,
        default=None,
        help="Specific n_estimators value to plot. Defaults to the best estimator by --best-metric.",
    )
    parser.add_argument(
        "--best-metric",
        default=DEFAULT_BEST_METRIC,
        help="Metric used to choose the plotted estimator when --plot-estimator is not set.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip Plotly report generation and only write the KPI CSV.",
    )
    parser.add_argument(
        "--allow-short",
        action="store_true",
        default=True,
        help="Allow the strategy to open short positions when the model predicts -1.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    for name, value in vars(args).items():
        print(f"{name}: {value}")

    timeframe = args.timeframe.lower().strip()
    resample = normalize_freq(args.resample) if args.resample else None
    freq = resample or timeframe_to_freq(timeframe)

    # loading the raw data
    dataframe = load_timeframe_data(timeframe, args.data_root)
    output_path = args.output or Path(f"KPI_{timeframe}.csv")

    # creating the various key value pairs that will be passed for backtesting
    # the hyperparameter being optimized for is the number of estimators, i.e the number of decision trees in the model
    kv_pairs_list = [
        {
            "dataframe": dataframe,
            "num_estimators": num_estimators,
            "freq": freq,
            "resample": resample,
            "allow_short": args.allow_short,
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

    if not args.no_plot:
        kpis_by_estimator = pd.DataFrame.from_dict(backtest_results, orient="index")
        plot_estimator = select_plot_estimator(
            kpis_by_estimator,
            args.plot_estimator,
            args.best_metric,
        )
        plot_result = run_model_backtest(
            dataframe,
            plot_estimator,
            freq,
            resample,
            args.allow_short,
        )
        plot_output_path = args.plot_output or default_plot_output_path(timeframe, resample)
        report_title = f"BTC-USD {timeframe} Backtest"
        if resample:
            report_title = f"{report_title} Resampled to {resample}"
        if not args.allow_short:
            report_title = f"{report_title} Long Only"
        write_plotly_report(plot_result, plot_output_path, report_title, plot_estimator)

        print(f"Saved Plotly report to {plot_output_path}")

    print(f"Saved KPI results to {output_path}")


if __name__ == "__main__":
    main()
