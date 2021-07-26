""" Module containing implementation of plotting operations.

Currently, we support interactive plotting in two variants: single figure and grid.
The plots contain a Slider widget that allows for easy switching between different figures, be it
the single figure or grid variant.
"""
from __future__ import annotations
import math
from enum import Enum
from typing import Callable, List, Tuple, Any

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import solution as sol


# TODO: the plotting is currently implemented only for the Sampling Problem
# Representation of (Project name, Workloads) grid
GridTiles = Tuple[str, List[str]]
# Plotting function protytpe
PlotCallable = Callable[[sol.SolutionSet], None]


# DataFrame columns
DF_DATA = ['project', 'workload', 'level', 'sampled', 'original']
# Number of plots in a grid, i.e., 2x2
GRID_SIZE = 4
# Default font size values
SUPTITLE_SIZE = 18
TITLE_SIZE = 16
LABEL_SIZE = 16
TICK_SIZE = 14


def plot_stub(_: sol.SolutionSet) -> None:
    """ A stub plotting function used when no plotting is requested.
    """
    pass


def plot_interactive(solutions: sol.SolutionSet) -> None:
    """ Plot results from the SolutionSet one figure at a time with an interactive Slider widget.

    :param solutions: a collection of solutions to plot 
    """
    def update_plot(val: int) -> None:
        """ Plot update function supplied as a callback to the Slider widget.

        :param val: the current Slider value
        """
        # Redraw the plot and re-configure the plot attributes
        draw_plot(dataset, solutions, ax, tiles[val])
        configure_plot(ax, tiles[val][0], **solutions.plot_params)

    # Initialize seaborn
    sns.set_theme()
    # Transform the solution set to a dataframe and split it into |slider interval| chunks 
    dataset = build_df(solutions)
    tiles = build_grid_tiles(dataset, 1)
    # Initial plot to obtain the axes that will be redrawn acording to the Slider value
    ax = sns.lineplot(
        x='level', y='call count', hue='call type', data=get_subframe(dataset, tiles[0])
    )
    draw_plot(dataset, solutions, ax, tiles[0])
    configure_plot(ax, tiles[0][0], **solutions.plot_params)

    # If there is more than one plot, also draw a Slider widget
    if len(tiles) > 1:
        axwl = plt.axes((0.25, 0.02, 0.5, 0.03))
        workload_slider = Slider(
            axwl, 'Workload ID', 0, len(tiles) - 1, valinit=0, 
            valstep=list(range(len(tiles))), valfmt='%d'
        )
        # Register the callback function
        workload_slider.on_changed(update_plot)
    plt.show()


def plot_interactive_grid(solutions: sol.SolutionSet) -> None:
    """ Plot results from the SolutionSet as a 2x2 grid with an interactive Slider widget.

    :param solutions: a collection of solutions to plot 
    """
    def update_plot(val: int) -> None:
        """ Plot update function supplied as a callback to the Slider widget.

        :param val: the current Slider value
        """
        # Update the grid plots based on the tiles index
        draw_grid(dataset, solutions, grid, tiles[val])
        configure_grid_plot(grid, tiles[val][0], **solutions.plot_params)

    # Initialize seaborn
    sns.set_theme()
    # Build a dataframe from the obtained solutions and split the data into 2x2 tiles
    dataset = build_df(solutions)
    tiles = build_grid_tiles(dataset)
    # Create the 2x2 subplot grid
    grid = sns.FacetGrid(data=get_subframe(dataset, tiles[0]), col='workload', col_wrap=2)
    # Initialize the grid 
    draw_grid(dataset, solutions, grid, tiles[0])
    configure_grid_plot(grid, tiles[0][0], **solutions.plot_params)

    # Add a slider to the Grid that switches between 2x2 workload sets
    if len(tiles) > 1:
        axwl = plt.axes((0.25, 0.03, 0.5, 0.03))
        workload_slider = Slider(
            axwl, 'Workload Set ID', 0, len(tiles) - 1, valinit=0, 
            valstep=list(range(len(tiles))), valfmt='%d'
        )
        workload_slider.on_changed(update_plot)
    # Show the grid
    plt.tight_layout()
    plt.show()


def build_df(solutions: sol.SolutionSet) -> pd.DataFrame:
    """ Build a DataFrame out of a solution set. The dataframe will be in a long format suitable
    for seaborn / matplotlib plotting. Furthermore, the dataframe rows will be sorted wrt the
    maximum number of calls in each level. 

    :param solutions: a collection of solutions to convert to a DataFrame

    :return: constructed and sorted DataFrame object
    """
    # Create a dataframe grouped by levels and focusing on maximum number of calls
    # I.e., for each level, we will record function with the most number of calls
    dfs = pd.concat([
        solution.to_dataframe().groupby('level').max().reset_index()[DF_DATA] 
        for solution in solutions], 
        ignore_index=True
    )
    # Categorical sorting: 
    #  - https://stackoverflow.com/questions/23482668/sorting-by-a-custom-list-in-pandas
    # Sort the workloads wrt largest Y values (number of calls) to ensure we plot similarly 
    # scaled plots in the grids
    sorter = (
        dfs.groupby(['workload'], as_index=False)['original']
        .max().sort_values('original')['workload'].tolist()
    )
    # Change the workload type to a category and set the ordering according to the sorter
    dfs.workload = dfs.workload.astype("category")
    dfs.workload.cat.set_categories(sorter, inplace=True)
    # Sort the records in the dataframe based on the workload ordering
    dfs.sort_values('workload', inplace=True, ignore_index=True)
    # Revert the category type back to plain string
    dfs.workload = dfs.workload.astype(str)
    # Convert the DF to a long format
    return dfs.melt(
        id_vars=['level', 'project', 'workload'], var_name='call type', value_name='call count'
    )


def build_grid_tiles(dataset: pd.DataFrame, size: int = GRID_SIZE) -> List[GridTiles]:
    """ Split the DataFrame content into grid tiles (e.g., 1x1, 2x2, 3x3, ...) that will be plotted 
    together based on the Slider value.

    :param dataset: the input Dataframe to split
    :param size: maximum number of tiles in each grid

    :return: a sequence of grids (project + workloads) 
    """
    seq = []
    # Get all projects in the dataset
    projects = dataset['project'].unique()
    for project in projects:
        # Get all workloads associated with the current project
        wl = list(dataset.loc[dataset['project'] == project]['workload'].unique())
        # Create a list of 'size' grid tiles for the given project
        seq.extend([(project, wl[i:i + size]) for i in range(0, len(wl), size)])
    return seq


def get_subframe(dataset: pd.DataFrame, tile: GridTiles) -> pd.DataFrame:
    """ Extract rows from the dataset that are matching the supplied grid tiles.

    :param dataset: the Dataframe to extract from
    :param tile: grid tile with identification of the project and workloads to extract

    :return: a subset of the dataset frame
    """
    # Obtain all rows from the dataset that are included in the tile
    return dataset.loc[(dataset['project'] == tile[0]) & (dataset['workload'].isin(tile[1]))]


def draw_grid(
    dataset: pd.DataFrame, solutions: sol.SolutionSet, grid: sns.FacetGrid, tiles: GridTiles
) -> None:
    """ Draw a grid plot with solutions specified by the tiles.

    :param dataset: dataframe of the whole solution set
    :param solutions: a collection of solutions
    :param grid: pre-configured grid plot structure
    :param tiles: specification of the solutions that should be plotted
    """
    # Extract subset of the dataframe that contains solutions from the tiles
    tile_frame = get_subframe(dataset, tiles)
    # Plot each solution
    for idx, workload in enumerate(tiles[1]):
        # Clear the ax configuration, plot and reconfigure the ax again
        # The axes must be cleared and reconfigured since the plots are being redrawn due to the
        # slider and old ax configuration visually breaks the plots
        grid.axes[idx].clear()
        sns.lineplot(
            x='level', y='call count', hue='call type', 
            data=tile_frame.loc[tile_frame['workload'] == workload], ax=grid.axes[idx]
        )
        grid.axes[idx].legend(loc='upper left', title='Call count')
        # Set a title for each plot
        grid.axes[idx].set_title(build_title(solutions[tiles[0], workload]), fontsize=TITLE_SIZE)


def configure_grid_plot(grid: sns.FacetGrid, title: str, **kwargs: Any) -> None:
    """ Configure the axes of a grid plot so that the grid plot is not too crammed.

    :param grid: the grid plot object
    :param title: a title of the whole grid, not the respective plots
    """
    for idx, ax in enumerate(grid.axes):
        # Disable plot ticks
        ax.tick_params(bottom=False, top=False, left=False, right=False, labelsize=TICK_SIZE)
        # Top-left plot
        if idx == 0:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("")
            ax.set_ylabel(ax.get_ylabel(), fontsize=LABEL_SIZE, weight='bold')
        # Top-right plot  
        elif idx == 1:
            ax.tick_params(labelbottom=False, labelleft=False)
            ax.set_xlabel("")
            ax.set_ylabel("")
        # Bottom-left plot
        elif idx == 2:
            ax.set_xlabel(ax.get_xlabel(), fontsize=LABEL_SIZE, weight='bold')
            ax.set_ylabel(ax.get_ylabel(), fontsize=LABEL_SIZE, weight='bold')  
        # Bottom-right plot
        elif idx == 3:
            ax.tick_params(labelleft=False)
            ax.set_ylabel("")
            ax.set_xlabel(ax.get_xlabel(), fontsize=LABEL_SIZE, weight='bold')
    # Grid title
    grid.fig.suptitle(f'{title} {kwargs.get("suptitle", "")}', fontsize=SUPTITLE_SIZE)


def draw_plot(
    dataset: pd.DataFrame, solutions: sol.SolutionSet, ax: plt.Axes, tile: GridTiles
) -> None:
    """ Draw a solution identified by the tile on a single plot.

    :param dataset: dataframe of the whole solution set
    :param solutions: a collection of solutions
    :param ax: an existing plot ax
    :param tile: identification of the solution that should be plotted
    """
    # Clear the ax configuration, plot and reconfigure the ax again
    # Same as in the grid drawing
    ax.clear()
    sns.lineplot(x='level', y='call count', hue='call type', data=get_subframe(dataset, tile), ax=ax)
    ax.legend(loc='upper left', title='Call count')
    # Set a title for the plot
    ax.set_title(build_title(solutions[tile[0], tile[1][0]]), fontsize=TITLE_SIZE)


def configure_plot(ax: plt.Axes, title: str, **kwargs: Any) -> None:
    """ Configure the ax of a plot.

    :param ax: an existing plot ax
    :param title: title of the whole figure
    """
    ax.tick_params(bottom=False, top=False, left=False, right=False, labelsize=TICK_SIZE)
    ax.set_xlabel(ax.get_xlabel(), fontsize=LABEL_SIZE, weight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontsize=LABEL_SIZE, weight='bold')
    plt.suptitle(f'{title} {kwargs.get("suptitle", "")}', fontsize=SUPTITLE_SIZE)


def build_title(solution: sol.S) -> str:
    """ Create a fitting plot title from the Solution attributes

    :param solution: a specific solution to create the title for

    :return: a LaTeX title string
    """
    # Mention each parameter in the title
    params_str = ''
    for name, value in solution.parameters.items():
        params_str += rf'[${name} = {value:.3f}$] '
    # Identify the plot by the workload name and set its resulting fitness
    return rf"W={solution.workload.name}  [$fitness = {solution.fitness:.3f}$] " + params_str


class PlotType(Enum):
    """ Enumeration of the possible plot types (e.g., None, Grid, Single, etc.) and mapping to
    the associated plotting functions.
    """
    GRID = 'grid'
    PLOT = 'plot'
    NONE = 'none'

    @staticmethod
    def supported() -> List[str]:
        """ Retrieve the supported plotting types as strings

        :return: a collection of plotting types
        """
        return [p.value for p in PlotType]

    @staticmethod
    def to_func(value: PlotType) -> PlotCallable:
        """ Plotting type string -> plotting function mapping.

        :param value: the plotting type

        :return: the plotting function
        """
        return {
            PlotType.GRID: plot_interactive_grid, 
            PlotType.PLOT: plot_interactive,
            PlotType.NONE: plot_stub    
        }[value]
