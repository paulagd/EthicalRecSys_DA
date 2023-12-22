import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from IPython import embed


# def bar_plot(ax, data, labels, colors=None, total_width=0.8, single_width=1, ylim=0.2, offsetx=0):
#     """Draws a bar plot with multiple bars per data point.
#
#     Parameters
#     ----------
#     ax : matplotlib.pyplot.axis
#         The axis we want to draw our plot on.
#
#     data: dictionary
#         A dictionary containing the data we want to plot. Keys are the names of the
#         data, the items is a list of the values.
#
#         Example:
#         data = {
#             "x":[1,2,3],
#             "y":[1,2,3],
#             "z":[1,2,3],
#         }
#
#     colors : array-like, optional
#         A list of colors which are used for the bars. If None, the colors
#         will be the standard matplotlib color cyle. (default: None)
#
#     total_width : float, optional, default: 0.8
#         The width of a bar group. 0.8 means that 80% of the x-axis is covered
#         by bars and 20% will be spaces between the bars.
#
#     single_width: float, optional, default: 1
#         The relative width of a single bar within a group. 1 means the bars
#         will touch eachother within a group, values less than 1 will make
#         these bars thinner.
#
#     legend: bool, optional, default: True
#         If this is set to true, a legend will be added to the axis.
#     """
#
#     # Check if colors where provided, otherwhise use the default color cycle
#     if colors is None:
#         #    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#         colors = ['grey', 'cadetblue']
#
#     # Number of bars per group
#     n_bars = len(data)
#
#     # The width of a single bar
#     bar_width = total_width / n_bars
#
#     # List containing handles for the drawn bars, used for the legend
#     bars = []
#
#     # Iterate over all data
#     for i, (name, values) in enumerate(data.items()):
#         # The offset in x direction of that bar
#         x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
#
#         # Draw a bar for every value of that type
#         for x, y in enumerate(values):
#             bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i])
#
#         # Add a handle to the last drawn bar, which we'll need for the legend
#         bars.append(bar[0])
#
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
#     ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
#
#     ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)
#     ax.set_xlabel("items", fontsize=16)
#     ax.set_ylabel("Merits", fontsize=16)
#     ax.grid(axis='x')
#     ax.tick_params(axis='x', size=5, rotation=45)
#     # ax.spines.right.set_visible(False)
#     # ax.spines.left.set_visible(False)
#     # ax.spines.top.set_visible(False)
#     plt.xlim(-0.5 + offsetx)
#     plt.ylim(0, ylim)
#
#     # Draw legend if we need
#     hand = []
#     for i, label in enumerate(labels):
#         hand.append(mlines.Line2D([], [], marker='o', linestyle='None', markersize=14, color=colors[i], label=label))
#
#     # plt.xticks(fontsize=5, rotation=45)
#     ax.legend(handles=hand, fontsize=14,
#               ncol=len(labels), title_fontsize=16, loc='best', facecolor='white', framealpha=1
#               )
#
#
# def get_mp(M1, M2, M3):
#     a = pd.DataFrame([M1, M2, M3]).T.reset_index()
#     a.rename(columns={"index": "items", 0: "M1", 1:"M2", 2:'M3'}, inplace=True)
#     a.sort_values(by=['M1'], ascending=False, inplace = True)
#     a.reset_index(drop=True, inplace=True)
#     return a
#
#
# def plot_merits(tr, val, test, ylim=0.16, topn=50):
#     a = get_mp(tr, val, test)
#     data = {
#         "train": list(a.M1.fillna(0))[:topn],
#         "val": list(a.M2.fillna(0))[:topn],
#         "test": list(a.M3.fillna(0))[:topn],
#     }
#
#     fig, ax = plt.subplots()
#     bar_plot(ax, data, ['train', 'val', 'test'], total_width=.8, single_width=1,
#              colors=['cadetblue', '#F2BC52', 'lightcoral'], ylim=ylim)
#     plt.savefig(f'data_merits_comparison.pdf', bbox_inches='tight')
#
#


# def plot_both(cov_dict, hr_dec, x2, y2, color_item, t0, t1, t2, title):
#
#     colors1 = ['teal', 'mediumturquoise', 'paleturquoise']
#     colors2 = [color_item[i] for i in x2]
#
#     y0 = [cov_dict[k] for k in colors1]
#     y1 = [hr_dec[k] for k in colors1]
#
#     fig, axs = plt.subplots(1, 3, figsize=(12, 5))
#     axs[0].bar(list(range(len(colors1))), y0, color=colors1)
#     axs[0].set_title(t0)
#     axs[0].set_ylim([0, 1])
#
#     axs[1].bar(list(range(len(colors1))), y1, color=colors1)
#     axs[1].set_title(t1)
#     axs[1].set_ylim([0, 1])
#
#
#     axs[2].bar(list(range(len(x2))), y2, color=colors2)
#     axs[2].set_title(t2)
#     plt.tight_layout()
#     plt.savefig(f'{title}.pdf')