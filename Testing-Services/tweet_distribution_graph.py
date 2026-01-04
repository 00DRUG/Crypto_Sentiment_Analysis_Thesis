import matplotlib.pyplot as plt
import numpy as np


data = {
    '2021-02': 44443,
    '2021-03': 4140,
    '2021-04': 58060,
    '2021-05': 21782,
    '2021-06': 125795,
    '2021-07': 466079,
    '2021-08': 488987,
    '2021-09': 23510,
    '2021-10': 351796,
    '2021-11': 359630,
    '2021-12': 55301,
    '2022-01': 260087,
    '2022-02': 79475,
    '2022-03': 360058,
    '2022-04': 417155,
    '2022-05': 356018,
    '2022-06': 346212,
    '2022-07': 193654,
    '2022-08': 41925,
    '2022-09': 182046,
    '2022-10': 146753,
    '2022-11': 202262,
    '2022-12': 43233,
    '2023-01': 60887,
    '2023-02': 63635,
    '2023-03': 106126
}


months = list(data.keys())
counts = list(data.values())


plt.figure(figsize=(14, 7))


bars = plt.bar(months, counts, color='#2c7bb6', zorder=3)


plt.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)


plt.ylabel('Number of Tweets', fontsize=12, labelpad=10)


plt.xticks(rotation=90, fontsize=10)


def format_k(value):
    if value >= 1000:
        return f'{int(value/1000)}k'
    return str(value)


for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 5000,
        format_k(height),
        ha='center', va='bottom',
        fontsize=9, rotation=0, color='black'
    )

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


plt.tight_layout()
plt.savefig('tweet_distribution.png', dpi=300)
plt.show()