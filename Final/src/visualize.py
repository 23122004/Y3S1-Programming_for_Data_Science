import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
from math import pi

from . import config

def setup_plot_style():
    plt.style.use(config.PLOT_STYLE)
    sns.set_palette(config.COLOR_PALETTE)
class Plotter:
    @staticmethod
    def _create_grid(n_plots: int, n_cols: int, figsize: Tuple = None,
                     default_col_width: int = 5, default_row_height: int = 4):
        n_rows = (n_plots + n_cols - 1) // n_cols
        if figsize is None:
            figsize = (default_col_width * n_cols, default_row_height * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows > 1 or n_cols > 1:
            axes = axes.ravel()
        
        return fig, axes, n_rows

    @staticmethod
    def _hide_unused_axes(axes, used_count: int):
        for idx in range(used_count, len(axes)):
            axes[idx].axis('off')

    @staticmethod
    def _finalize():
        plt.tight_layout()
        plt.show()

    @staticmethod
    def histograms(df: pd.DataFrame,
                   columns: Optional[List[str]] = None,
                   n_cols: int = 4,
                   figsize: Tuple = None,
                   bins: int = 30):
        setup_plot_style()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        fig, axes, _ = Plotter._create_grid(len(columns), n_cols, figsize)

        for idx, col in enumerate(columns):
            if idx < len(axes):
                df[col].hist(bins=bins, ax=axes[idx], edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}', fontsize=10, fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')

        Plotter._hide_unused_axes(axes, len(columns))
        Plotter._finalize()

    @staticmethod
    def boxplots(df: pd.DataFrame,
                 columns: List[str],
                 n_cols: int = 4,
                 figsize: Tuple = None):
        setup_plot_style()
        if not columns:
            print("Không có cột nào!")
            return

        fig, axes, _ = Plotter._create_grid(len(columns), n_cols, figsize, default_row_height=5)

        for idx, col in enumerate(columns):
            sns.boxplot(y=df[col], ax=axes[idx])
            axes[idx].set_title(f'Box Plot of {col}', fontsize=10, fontweight='bold')
            axes[idx].set_ylabel(col)

        Plotter._hide_unused_axes(axes, len(columns))
        Plotter._finalize()

    @staticmethod
    def heatmap(data: pd.DataFrame,
                figsize: Tuple = config.FIG_SIZE_LARGE,
                annot: bool = True,
                cmap: str = config.HEATMAP_CMAP,
                title: str = 'Heatmap',
                fmt: str = '.2f'):
        plt.figure(figsize=figsize)
        sns.heatmap(data, annot=annot, fmt=fmt,
                   cmap=cmap, center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        Plotter._finalize()

    @staticmethod
    def scatter(df: pd.DataFrame, x: str, y: str,
                title: str = None,
                xlabel: str = None,
                ylabel: str = None,
                figsize: Tuple = config.FIG_SIZE_SMALL,
                alpha: float = 0.6,
                color: str = 'steelblue',
                line_color: str = None):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if line_color:
            sns.regplot(data=df, x=x, y=y,
                        scatter_kws={'alpha': 0.6, 'color': color},
                        line_kws={'color': line_color})

        ax.scatter(df[x], df[y], alpha=alpha, color=color)
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        ax.set_title(title or f'{x} vs {y}', fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)
        Plotter._finalize()

    @staticmethod
    def boxplot(df: pd.DataFrame,
                x: str = None,
                y: str = None,
                title: str = None,
                xlabel: str = None,
                ylabel: str = None,
                palette: str = 'viridis',
                figsize: Tuple = config.FIG_SIZE_SMALL):
        plt.figure(figsize=figsize)
        if x:
            sns.boxplot(data=df, x=x, y=y, palette=palette, hue=x, legend=False)
        else:
            sns.boxplot(data=df, y=y)
        plt.title(title or (f'{y} by {x}' if x else f'Distribution of {y}'), fontsize=14)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()

    @staticmethod
    def multi_boxplots(df: pd.DataFrame,
                       x: str,
                       y_cols: List[str],
                       titles: List[str] = None,
                       ylabels: List[str] = None,
                       palettes: List[str] = None,
                       figsize: Tuple = config.FIG_SIZE_WIDE):
        n_plots = len(y_cols)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)

        if n_plots == 1:
            axes = [axes]

        titles = titles or [f'{y} by {x}' for y in y_cols]
        ylabels = ylabels or y_cols
        palettes = palettes or ['viridis'] * n_plots

        for idx, (y, title, ylabel, palette) in enumerate(zip(y_cols, titles, ylabels, palettes)):
            sns.boxplot(data=df, x=x, y=y, ax=axes[idx],
                       hue=x, palette=palette, legend=False)
            axes[idx].set_title(title)
            axes[idx].set_ylabel(ylabel)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def dual_axis(df: pd.DataFrame,
                  x: str,
                  y1: str,
                  y2: str,
                  title: str = None,
                  y1_label: str = None,
                  y2_label: str = None,
                  y1_color: str = 'tab:blue',
                  y2_color: str = 'tab:red',
                  figsize: Tuple = config.FIG_SIZE_MEDIUM,
                  agg_func: str = 'mean'):
        agg_data = df.groupby(x, observed=True)[[y1, y2]].agg(agg_func).reset_index()

        fig, ax1 = plt.subplots(figsize=figsize)

        sns.barplot(data=agg_data, x=x, y=y1, ax=ax1, alpha=0.6, color=y1_color)
        ax1.set_ylabel(y1_label or f'Average {y1}', color=y1_color, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=y1_color)

        ax2 = ax1.twinx()
        sns.lineplot(data=agg_data, x=x, y=y2, ax=ax2, marker='o',
                    color=y2_color, linewidth=2.5, sort=False)
        ax2.set_ylabel(y2_label or f'Average {y2}', color=y2_color, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=y2_color)

        plt.title(title or f'{y1} vs {y2} by {x}', fontsize=14)
        plt.grid(axis='x', linestyle='--')
        plt.show()

    @staticmethod
    def barplot(df: pd.DataFrame,
                x: str,
                y: str,
                title: str = None,
                xlabel: str = None,
                ylabel: str = None,
                palette: str = 'viridis',
                figsize: Tuple = config.FIG_SIZE_MEDIUM,
                show_values: bool = False):
        plt.figure(figsize=figsize)
        ax = sns.barplot(data=df, x=x, y=y, palette=palette, hue=x, legend=False)

        if show_values:
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', fontsize=9)

        plt.title(title or f'{y} by {x}', fontsize=14)
        plt.xlabel(xlabel or x)
        plt.ylabel(ylabel or y)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.show()

    @staticmethod
    def radar(data: pd.DataFrame,
              features: List[str],
              title: str = 'Radar Chart',
              figsize: Tuple = (10, 10),
              colors: List[str] = None):
        
        N = len(features)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        plt.figure(figsize=figsize)
        ax = plt.subplot(111, polar=True)

        n_items = len(data)
        if colors is None:
            colors = sns.color_palette("husl", n_items)

        # Min-Max scaling
        data_norm = (data[features] - data[features].min()) / \
                    (data[features].max() - data[features].min())

        for idx, (label, row) in enumerate(data_norm.iterrows()):
            values = row.values.flatten().tolist()
            values += values[:1]
            color = colors[idx % len(colors)]
            ax.plot(angles, values, linewidth=2, linestyle='solid',
                    label=f'{label}', color=color)
            ax.fill(angles, values, color=color, alpha=0.2)

        plt.xticks(angles[:-1], features, size=12)
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1.0], ["Low", "Med", "High", "Max"], color="grey", size=10)
        plt.ylim(0, 1.1)
        plt.title(title, size=16, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.show()

    @staticmethod
    def grouped_bars(data: Dict,
                     value_key: str,
                     title: str = 'Grouped Bar Chart',
                     xlabel: str = 'Features',
                     ylabel: str = 'Value',
                     figsize: Tuple = config.FIG_SIZE_SMALL,
                     colors: Dict = None):
        groups = list(data.keys())
        features = list(data[groups[0]][value_key].keys())

        x = np.arange(len(features))
        width = 0.8 / len(groups)
        fig, ax = plt.subplots(figsize=figsize)

        if colors is None:
            colors = {}

        for i, group in enumerate(groups):
            values = [data[group][value_key][f] for f in features]
            offset = width * (i - len(groups)/2 + 0.5)
            bars = ax.bar(x + offset, values, width, label=group,
                         color=colors.get(group, f'C{i}'), alpha=0.8)

            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        Plotter._finalize()

    @staticmethod
    def metrics_bars(data: Dict,
                     metrics: List[str] = None,
                     labels: List[str] = None,
                     title: str = 'Metrics Comparison',
                     figsize: Tuple = config.FIG_SIZE_SMALL,
                     colors: Dict = None):
        
        groups = list(data.keys())
        metrics = metrics or ['r2_score', 'rmse', 'mae']
        labels = labels or metrics

        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]

        if colors is None:
            colors = {}

        for idx, (metric, label) in enumerate(zip(metrics, labels)):
            values = [data[g]['metrics'][metric] for g in groups]
            bars = axes[idx].bar(groups, values,
                                color=[colors.get(g, f'C{i}') for i, g in enumerate(groups)],
                                alpha=0.8, edgecolor='black')

            for bar, val in zip(bars, values):
                axes[idx].annotate(f'{val:.4f}',
                                  xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                  xytext=(0, 3), textcoords="offset points",
                                  ha='center', va='bottom', fontsize=10, fontweight='bold')

            axes[idx].set_title(label, fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(label)
            axes[idx].grid(axis='y', alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight='bold')
        Plotter._finalize()

class SHAPPlotter:   
    def __init__(self, model):
        import shap
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = None

    def compute_shap_values(self, X: pd.DataFrame):
        self.shap_values = self.explainer.shap_values(X)
        return self.shap_values

    def plot_summary(self, X: pd.DataFrame, title: str = 'SHAP Summary'):
        import shap
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=config.FIG_SIZE_MEDIUM)
        shap.summary_plot(self.shap_values, X, show=False)
        plt.title(title, fontsize=14)
        plt.show()

    def plot_dependence(self, X: pd.DataFrame, feature: str,
                        interaction_feature: str, title: str = 'SHAP Dependence'):
        import shap
        if self.shap_values is None:
            self.compute_shap_values(X)

        plt.figure(figsize=config.FIG_SIZE_SMALL)
        shap.dependence_plot(feature, self.shap_values, X,
                            interaction_index=interaction_feature, show=False)
        plt.title(title, fontsize=14)
        plt.show()
