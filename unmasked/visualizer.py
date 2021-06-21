from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import sem, t

from unmasked import configs
from unmasked.utils import get_legend_label

SHOW_PARTIAL_FIGURE = True  # whether to show partially completed figure while making large figure

MULTI_AXIS_LEG_NUM_COLS = 2  # 2 or 3 depending on space
MULTI_AXIS_LEG_OFFSET = 0.12
STANDALONE_LEG_OFFSET = 0.45
STANDALONE_FIG_SIZE = (4, 4)

# lines figure
Y_TICK_LABEL_FONTSIZE = 5


def make_ax_title(phenomenon: str,
                  paradigm: str,
                  ) -> str:
    name = f'{phenomenon}\n{paradigm}'

    ax_title = name.replace('_', ' ')
    ax_title = ax_title.replace('coordinate', 'coord.')
    ax_title = ax_title.replace('structure', 'struct.')
    ax_title = ax_title.replace('prepositional', 'prep.')
    ax_title = ax_title.replace('agreement', 'agreem.')
    ax_title = ax_title.replace('determiner', 'det.')
    return ax_title


@dataclass
class ParadigmData:
    """data for a single paradigm"""
    phenomenon: str
    paradigm: str
    group_name2accuracies: Dict[str, np.array]  # accuracies corresponding to replications, for each group


class Visualizer:
    def __init__(self,
                 num_paradigms: int,
                 group_names: List[str],
                 y_lims: Optional[List[float]] = None,
                 fig_size: int = (6, 5),
                 dpi: int = 300,
                 show_partial_figure: bool = SHOW_PARTIAL_FIGURE,
                 confidence: float = 0.90,
                 ):

        self.group_names = group_names
        self.num_groups = len(group_names)
        self.labels = get_legend_label(group_names)
        self.y_lims = y_lims or [50, 101]
        self.show_partial_figure = show_partial_figure
        self.confidence = confidence

        # calc num rows needed
        self.num_cols = 5
        num_paradigms_and_summary = num_paradigms + 1
        num_rows_for_data = num_paradigms_and_summary / self.num_cols
        num_rows_for_legend = 1
        self.num_rows = int(num_rows_for_data) + num_rows_for_legend
        self.num_rows += 1 if not num_rows_for_data.is_integer() else 0  # to fit summary

        self.fig, self.ax_mat = plt.subplots(self.num_rows, self.num_cols,
                                             figsize=fig_size,
                                             dpi=dpi,
                                             )

        self.y_ticks = [50, 60, 70, 80, 90, 100]

        # remove all tick labels ahead of plotting to reduce space between subplots
        for ax in self.ax_mat.flatten():
            # y-axis
            y_ticks = []
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks, fontsize=configs.Figs.tick_font_size)
            # x-axis
            ax.set_xticks([])
            ax.set_xticklabels([])

        self.axes_for_legend = self.ax_mat[-1]
        self.axes = enumerate(ax for ax in self.ax_mat.flatten())
        self.pds = []  # data, one for each axis/paradigm

        self.y_axis_label = 'Accuracy'

        self.width = 0.2  # of bar

    def update(self,
               pd: ParadigmData,
               ) -> None:
        """draw plot on one axis, corresponding to one paradigm"""

        self.pds.append(pd)

        # get next axis
        ax_id, ax = next(self.axes)

        # title
        ax_title = make_ax_title(pd.phenomenon, pd.paradigm)
        ax.set_title(ax_title, fontsize=configs.Figs.title_font_size)

        # y axis
        if ax_id % self.ax_mat.shape[1] == 0:
            ax.set_ylabel(self.y_axis_label, fontsize=configs.Figs.ax_font_size)
            y_ticks = [50, 60, 70, 80, 90, 100]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks, fontsize=configs.Figs.tick_font_size)
        # x-axis
        ax.set_xticks([])
        ax.set_xticklabels([])

        # axis
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(self.y_lims)

        edges = [self.width * i for i in range(self.num_groups)]  # distances between x-ticks and bar-center
        colors = [f'C{i}' for i in range(self.num_groups)]

        x = np.arange(1)

        # plot
        for edge, color, group_name in zip(edges, colors, self.group_names):
            accuracies = pd.group_name2accuracies[group_name]
            y = np.mean(accuracies, axis=0)  # take average across reps

            # margin of error
            n = len(accuracies)
            h = sem(accuracies, axis=0) * t.ppf((1 + self.confidence) / 2, n - 1)  # margin of error

            # plot all bars belonging to a single model group (same color)
            ax.bar(x + edge,
                   y,
                   self.width,
                   yerr=h,
                   color=color,
                   zorder=3,
                   )

        # plot legend only once to prevent degradation in text quality due to multiple plotting
        if ax_id == 0:
            self._plot_legend(offset_from_bottom=MULTI_AXIS_LEG_OFFSET+0.02)

        if self.show_partial_figure:
            self.fig.tight_layout()
            self.fig.show()

    def plot_summary(self):
        """plot average accuracy (across all paradigms) in last axis"""

        # get next axis in multi-axis figure and plot summary there
        ax_id, ax = next(self.axes)
        self._plot_summary_on_axis(ax, label_y_axis=ax_id % self.ax_mat.shape[1] == 0, use_title=True)

        # remove axis decoration from any remaining axis
        for ax_id, ax in self.axes:
            ax.axis('off')

        # also plot boxplot summary in standalone figure
        fig_standalone, (ax1, ax2) = plt.subplots(2, figsize=STANDALONE_FIG_SIZE, dpi=300)
        self._plot_boxplot_summary_standalone(ax1)
        ax2.axis('off')
        fig_standalone.subplots_adjust(top=0.1, bottom=0.01)
        self._plot_legend(offset_from_bottom=STANDALONE_LEG_OFFSET + 0.02, fig=fig_standalone)

        # show
        self.fig.show()
        fig_standalone.show()

    def _plot_summary_on_axis(self,
                              ax: plt.axis,
                              label_y_axis: bool,
                              use_title: bool,
                              ):
        """used to plot summary on multi-axis figure"""

        # axis
        if use_title:
            ax.set_title('Average', fontsize=configs.Figs.title_font_size)
            y_axis_label = self.y_axis_label
        else:
            y_axis_label = f'Average {self.y_axis_label}'
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(self.y_lims)

        # x-axis
        ax.set_xticks([])
        ax.set_xticklabels([])

        # y axis
        if label_y_axis:
            ax.set_ylabel(y_axis_label, fontsize=configs.Figs.ax_font_size)
            ax.set_yticks(self.y_ticks)
            ax.set_yticklabels(self.y_ticks, fontsize=configs.Figs.tick_font_size)
        else:
            y_ticks = []
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks, fontsize=configs.Figs.tick_font_size)

        edges = [self.width * i for i in range(self.num_groups)]  # distances between x-ticks and bar-center
        colors = [f'C{i}' for i in range(self.num_groups)]
        x = np.arange(1)

        # plot
        for edge, color, group_name in zip(edges, colors, self.group_names):
            accuracies = np.array([pd.group_name2accuracies[group_name].mean().item() for pd in self.pds])
            y = accuracies.mean()

            # margin of error
            n = len(accuracies)
            h = sem(accuracies, axis=0) * t.ppf((1 + self.confidence) / 2, n - 1)  # margin of error

            # plot all bars belonging to a single model group (same color)
            ax.bar(x + edge,
                   y,
                   self.width,
                   yerr=h,
                   color=color,
                   zorder=3,
                   )

    def _plot_boxplot_summary_standalone(self,
                                         ax: plt.axis,
                                         ):
        """used to plot summary in standalone figure"""

        # axis
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([0.0, 1.0])

        # x axis
        x_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ax.set_xlabel('Average Accuracy', fontsize=configs.Figs.tick_font_size)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks, fontsize=configs.Figs.tick_font_size)

        # chance level
        ax.axvline(x=50, linestyle='dotted', color='grey')

        # boxplot data
        boxplot_data = []
        for group_name in self.group_names:
            data_for_one_group = [pd.group_name2accuracies[group_name].mean().item() for pd in self.pds]
            boxplot_data.append(data_for_one_group)

        # boxplot plots IQR and line at median, not mean
        positions = [n for n, _ in enumerate(self.group_names)][::-1]
        box_plot = ax.boxplot(boxplot_data,
                              positions=positions,
                              vert=False,
                              patch_artist=True,
                              medianprops={'color': 'black'},
                              flierprops={'markersize': 2},
                              boxprops=dict(linewidth=1),
                              zorder=2,
                              showfliers=False,
                              )
        # mark the mean
        means = [np.mean(x) for x in boxplot_data]
        ax.scatter(means, positions, zorder=3, color='black', s=5)

        # color
        colors = [f'C{i}' for i in range(self.num_groups)]
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        # remove y-axis
        ax.set_yticks([])
        ax.set_yticklabels([])

    def _plot_legend(self,
                     offset_from_bottom: float,
                     fig: Optional[plt.Figure] = None,
                     ):

        if fig is None:
            fig = self.fig

        labels = self.labels
        legend_elements = [Line2D([0], [0], color=f'C{n}', label=label) for n, label in enumerate(labels)]

        for ax in self.axes_for_legend:
            ax.axis('off')

        # legend
        fig.legend(handles=legend_elements,
                   loc='upper center',
                   bbox_to_anchor=(0.5, offset_from_bottom),
                   ncol=3,
                   frameon=False,
                   fontsize=configs.Figs.leg_font_size)
