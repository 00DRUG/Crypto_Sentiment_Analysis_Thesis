import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
matplotlib.use("TkAgg")
df = pd.read_csv("../Databases/btcind.csv", parse_dates=["Date"])


plt.figure(figsize=(14,6))
plt.plot(df["Date"], df["volume"], label="Open", alpha=0.7)


plt.title("Bitcoin volume")
plt.xlabel("Date")
plt.ylabel("Price in USD")
plt.legend()
plt.grid(True)
plt.show()
