import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas
pandas.set_option('display.max_columns', 50)
pandas.set_option('display.max_rows', 100)
pandas.set_option('display.width', 1000)

class DataGetter:
    def __init__(self, asset="ETH-USD", ref="BTC-USD",
                 start_date="2017-11-09", end_date="2024-03-04",
                 freq="1d"):
        self.asset = asset
        self.ref = ref
        self.start_date =start_date
        self.end_date =end_date
        self.freq = freq

        self.timeFrames = [1, 2, 7, 14, 21, 31]
        self.frame = None
        self.data = self.getData()
        self.scaler = StandardScaler()
        self.scaler.fit(self.data[:, :])
        self.scaledData = self.scaleData()


    def getData(self):
        df     = yf.download([self.asset], start=self.start_date, end=self.end_date, interval=self.freq)
        df_ref = yf.download([self.ref], start=self.start_date, end=self.end_date, interval=self.freq)
        print(df.head())
        print(df.shape)
        print(df_ref.head())
        print(df_ref.shape)

        # Features:
        # 1. Price and Volume Changes
        for i in self.timeFrames:
            df[f"pc-{i}"] = df["Adj Close"].pct_change(i)
            df[f"vc-{i}"] = df["Volume"].pct_change(i)
            df_ref[f"ref_pc-{i}"] = df_ref["Adj Close"].pct_change(i)
            df_ref[f"ref_vc-{i}"] = df_ref["Volume"].pct_change(i)



        # 2. Range of the price today
        df["Range"] = df["High"] - df["Low"]

        # 3. Moving Averages over multiple periods
        # 4. Exponential Moving Average
        for i in self.timeFrames:
            if i == 1: continue
            df[f"MA-{i}"] = df["Adj Close"].rolling(window=i).mean()
            df[f"EMA-{i}"] = df["Adj Close"].ewm(span=i, adjust=False).mean()

        # 5. Moving Average Convergence Divergence
        df["MACD"] = df["EMA-14"] - df["EMA-31"]
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # 6. RSI: Relative Strength Index.
        delta = df["pc-1"]
        up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
        roll_up, roll_down = up.rolling(window=14).mean(), down.rolling(window=14).mean()
        RS = roll_up / roll_down
        df["RSI"] = 100.0 - (100.0 / (1.0 + RS))

        # 7. Bollinger Bands
        df["MA_20"] = df["pc-1"].rolling(window=20).mean()
        df["STD_20"] = df["pc-1"].rolling(window=20).std()
        df["Upper_BB"] = df["MA_20"] + (df["STD_20"] * 2)
        df["Lower_BB"] = df["MA_20"] - (df["STD_20"] * 2)

        # 8. Volatility
        for i in self.timeFrames:
            if i == 1: continue
            df[f'volatility-{i}']         = np.log(1 + df[f"pc-1"]).rolling(i).std()

        # 9. Merge in our reference BTC data:
        df = df.merge(df_ref[[f"ref_pc-{i}" for i in self.timeFrames] + [f"ref_vc-{i}" for i in self.timeFrames]],
                      how="left", right_index=True, left_index=True)

        # Drop useless columns
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)

        self.frame = df
        # Clean NaN data:
        df.dropna(inplace=True)
        print(df.head())
        res = np.array(df)
        print(res.shape)
        self.priceArray = df["Adj Close"].to_numpy()
        self.dateArray = df.index.strftime('%Y-%m-%d').to_numpy()

        return res

    def scaleData(self):
        return StandardScaler().fit_transform(self.data[:, :])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, col_idx=None):
        if col_idx is None:
            return self.data[idx]
        elif col_idx < len(list(self.data.columns)):
            return self.data[idx][col_idx]
        else:
            raise IndexError