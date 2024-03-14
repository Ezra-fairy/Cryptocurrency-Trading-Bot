import numpy as np
import pandas
import yfinance as yf
from sklearn.preprocessing import StandardScaler
pandas.set_option('display.max_columns', None)
class DataGetter:
    """
    The class for getting data for assets.
    """

    def __init__(self, asset="BTC-USD", start_date=None, end_date=None, freq="1d",
                 timeframes=[1, 2, 5, 10, 20, 40]):
        self.asset = asset
        self.sd = start_date
        self.ed = end_date
        self.freq = freq

        self.timeframes = timeframes
        self.getData()

        self.scaler = StandardScaler()
        self.scaler.fit(self.data[:, 1:])

    def getData(self):

        asset = self.asset
        if self.sd is not None and self.ed is not None:
            print("1")
            df = yf.download([asset], start=self.sd, end=self.ed, interval=self.freq)
            df_spy = yf.download(["BTC-USD"], start=self.sd, end=self.ed, interval=self.freq)
        elif self.sd is None and self.ed is not None:
            print("2")
            df = yf.download([asset], end=self.ed, interval=self.freq)
            df_spy = yf.download(["BTC-USD"], end=self.ed, interval=self.freq)
        elif self.sd is not None and self.ed is None:
            print("3")
            df = yf.download([asset], start=self.sd, interval=self.freq)
            df_spy = yf.download(["BTC-USD"], start=self.sd, interval=self.freq)
        else:
            print("4")
            df = yf.download([asset], period="max", interval=self.freq)
            df_spy = yf.download(["BTC-USD"], interval=self.freq)

        # Reward - Not included in Observation Space.
        # Computes the fractional change from the immediately previous row
        df["rf"] = df["Adj Close"].pct_change().shift(-1)
        # Returns and Trading Volume Changes
        for i in self.timeframes:
            df_spy[f"spy_ret-{i}"] = df_spy["Adj Close"].pct_change(i)
            df_spy[f"spy_v-{i}"] = df_spy["Volume"].pct_change(i)

            df[f"r-{i}"] = df["Adj Close"].pct_change(i)
            df[f"v-{i}"] = df["Volume"].pct_change(i)


        # Volatility, standard deviation of how prices change over a certain period
        for i in [5, 10, 20, 40]:
            df[f'sig-{i}']         = np.log(1 + df["r-1"]).rolling(i).std()
            df_spy[f'spy_sig-{i}'] = np.log(1 + df_spy["spy_v-1"]).rolling(i).std()

        # Moving Average Convergence Divergence (MACD) to see momentum behind price changes
        df["macd_lmw"] = df["r-1"].ewm(span=26, adjust=False).mean()
        df["macd_smw"] = df["r-1"].ewm(span=12, adjust=False).mean()
        df["macd_bl"] = df["r-1"].ewm(span=9, adjust=False).mean()
        df["macd"] = df["macd_smw"] - df["macd_lmw"]

        # Relative Strength Indicator (RSI)
        rsi_lb = 5
        pos_gain = df["r-1"].where(df["r-1"] > 0, 0).ewm(rsi_lb).mean()
        neg_gain = df["r-1"].where(df["r-1"] < 0, 0).ewm(rsi_lb).mean()
        rs = np.abs(pos_gain / neg_gain)
        df["rsi"] = 100 * rs / (1 + rs)

        # Bollinger Bands
        bollinger_lback = 10
        df["bollinger"] = df["r-1"].ewm(bollinger_lback).mean()
        df["low_bollinger"] = df["bollinger"] - 2 * df["r-1"].rolling(bollinger_lback).std()
        df["high_bollinger"] = df["bollinger"] + 2 * df["r-1"].rolling(bollinger_lback).std()

        # SP500
        df = df.merge(df_spy[[f"spy_ret-{i}" for i in self.timeframes] + [f"spy_sig-{i}" for i in [5, 10, 20, 40]]],
                      how="left", right_index=True, left_index=True)

        # Filtering
        # df.interpolate('linear', limit_direction='both', inplace=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        self.frame = df
        self.data = np.array(df.iloc[:, 6:])
        print(df)
        print(df.shape)
        return

    def scaleData(self):
        self.scaled_data = self.scaler.fit_transform(self.data[:, 1:])
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, col_idx=None):
        if col_idx is None:
            return self.data[idx]
        elif col_idx < len(list(self.data.columns)):
            return self.data[idx][col_idx]
        else:
            raise IndexError


if __name__ == '__main__':
    asset = DataGetter()