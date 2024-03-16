import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from matplotlib.lines import Line2D


def drawLineGraph(dates, prices, actions, file_name):

    # Convert date strings to datetime objects
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

    # Plotting
    fig, ax = plt.subplots(figsize=(50, 30))
    ax.plot(dates, prices, label='Price', color='black', marker='o', linestyle='-')

    # Convert to Matplotlib date format
    dates = mdates.date2num(dates)

    # Mark buy and sell actions
    for i, (date, price, action) in enumerate(zip(dates, prices, actions)):
        if action == 1:  # Buy
            ax.scatter(date, price, color='red', label='Buy' if i == 0 else "", zorder=5, marker='^', s=300)
        elif action == -1:  # Sell
            ax.scatter(date, price, color='green', label='Sell' if i == 0 else "", zorder=5, marker='v', s=300)

    # Formatting Date
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Price Action Over Time")
    # Create custom legend
    legend_elements = [
        Line2D([0], [0], color='black', marker='o', linestyle='-', label='Price'),
        Line2D([0], [0], marker='^', color='w', label='Buy', markerfacecolor='red', markersize=5),
        Line2D([0], [0], marker='v', color='w', label='Sell', markerfacecolor='green', markersize=5),
    ]

    # Set legend with custom handles
    ax.legend(handles=legend_elements)
    plt.grid(True)

    # Rotate date labels
    plt.gcf().autofmt_xdate()
    plt.savefig(file_name)
    # plt.show()


