import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
matplotlib.use("TkAgg")
df = pd.read_csv("../Databases/btcind.csv", parse_dates=["Date"])

# OHLC4
df["OHLC4"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

#print(df[["Date", "open", "high", "low", "close", "OHLC4"]].head())

plt.figure(figsize=(14,6))
plt.plot(df["Date"], df["open"], label="Open", alpha=0.7)
plt.plot(df["Date"], df["close"], label="Close", alpha=0.7)
plt.plot(df["Date"], df["OHLC4"], label="OHLC4 (avg)", linewidth=2, color="black")

plt.xlabel("Date")
plt.ylabel("Price in USD")
plt.legend()
plt.grid(True)
plt.show()
