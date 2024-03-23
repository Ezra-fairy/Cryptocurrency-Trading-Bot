import random

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from matplotlib.lines import Line2D


def draw_action_graph(dates, prices, actions, file_name):

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




def plot_episode_values(losses, file_name, y_name, decimal):
    plt.style.use('seaborn-darkgrid')  # Use a nice style for the plot
    plt.figure(figsize=(20, 6))  # Set the figure size

    for i, loss in enumerate(losses):
        if i % 8 == 0:  # Only annotate every 5th point
            if decimal:
                loss_val = f"{loss:.2f}"
            else:
                loss_val = f"{loss:.0f}"
            plt.text(i, loss + max(losses)*0.02, loss_val, ha='center', va='bottom', fontsize=18, fontweight='bold', color='darkred')
    plt.plot(losses, label=f'{y_name} per Episode', color='teal', linewidth=2, marker='o', markersize=4, markerfacecolor='black', markeredgewidth=2, markeredgecolor='black')
    plt.title(f'{y_name} per Episode', fontsize=25, fontweight='bold', color='navy')
    plt.xlabel('Episodes', fontsize=20, fontweight='bold')
    plt.ylabel(y_name, fontsize=20, fontweight='bold')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()  # Adjust the layout to make room for the elements
    plt.savefig(file_name)


if __name__ == '__main__':
    capital_values = [random.randint(100000, 10000000) for _ in range(150)]
    plot_episode_values(capital_values, "test.png", "capital", False)