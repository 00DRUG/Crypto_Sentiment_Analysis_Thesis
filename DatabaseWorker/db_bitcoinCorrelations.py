import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

df = pd.read_csv("../Databases/btcind.csv", parse_dates=["Date"])


numeric_cols = ["open", "high", "low", "close", "volume", "OHLC4",
                "SMA_7", "EMA_7", "SMA_20", "EMA_20", "daily_return_pct"]


numeric_cols = [c for c in numeric_cols if c in df.columns]

df_num = df[numeric_cols].dropna()


sns.set(style="ticks", font_scale=1)
pairplot = sns.pairplot(df_num, diag_kind="kde", plot_kws={"alpha":0.6, "s":20, "edgecolor":"k"})
pairplot.fig.suptitle("Scatter matrix & distributions (right-skewed evident)", y=1.02)

plt.show()
