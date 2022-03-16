import os
import csv
from typing import Dict, Generator, List, Tuple

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import utils.values as values


FIG_DIR = os.path.join(values.RESULTS_DIR, 'figures')
FIG_SUF = '.pdf'
SAMPLE = 10


def pair_runs(run_names: List[str]) -> Generator[Tuple[str, str], None, None]:
    comma_runs = []
    plus_runs = set()
    for name in run_names:
        comma_idx = name.find(',')
        if comma_idx != -1:
            comma_runs.append((name, comma_idx))
        else:
            plus_runs.add(name)
    
    for run_name, comma_pos in comma_runs:
        plus_run = run_name[:comma_pos] + '+' + run_name[comma_pos + 1:]
        if plus_run not in plus_runs:
            raise RuntimeError(f'Matching run for {run_name} not found?')
        else:
            yield run_name, plus_run


def split_fitness(run_names: List[str]) -> Generator[Tuple[str, List[str]], None, None]:
    fitness_group: Dict[str, List[str]] = {}
    for name in run_names:
        fitness = name.split('_')[1]
        fitness_group.setdefault(fitness, []).append(name)
    for grp, names in fitness_group.items():
        yield grp, names 



# mixed = pd.read_csv('evo_data2.csv', delimiter=',')
# fixed = pd.read_csv('evo_data.csv', delimiter=',')
# mixed_good = mixed[mixed['name'].apply(lambda x: ',' not in x)]
# fixed_all = pd.concat([fixed, mixed_good])
# print(list(fixed_all['name'].unique()))

# fixed_all.to_csv('evo_data_fixed.csv', index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
# exit(1)

sns.set_theme()

plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['font.weight'] = 'bold'

df = pd.read_csv(os.path.join(values.RESULTS_DIR, 'data', 'evo_data.csv'))
df.rename(columns={'name': 'strategy'}, inplace=True)

# print(evo_dataset)
df_long = df.melt(
    id_vars=['strategy', 'generation', 'round', 'goal', 'fitness', 'strength'], var_name='ratio_type', value_name='ratio_values'
)

for goal in list(df_long['goal'].unique()):
    print(f"Goal: {goal}")
    df_long_goal = df_long[df_long['goal'] == goal]

    # Per-goal boxplots
    df_boxplot = df_long_goal.copy()
    df_boxplot[['strategy', 'fitness_type']] = df_boxplot['strategy'].str.split('_', 1, expand=True)
    df_boxplot = df_boxplot.groupby(['strategy', 'fitness_type', 'round', 'goal'], as_index=False)['fitness'].max()
    # df_boxplot.reset_index()
    print(df_boxplot)
    # max_fit_idx = df_boxplot.groupby(['strategy', 'fitness_type', 'round', 'goal'])['fitness'].transform(max) == df_boxplot['fitness']
    # print(df_boxplot[max_fit_idx])
    
    grid = sns.catplot(x='strategy', y="fitness", kind="box", col='fitness_type', data=df_boxplot)    
    for ax in grid.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, fontweight='bold')
    plt.suptitle(f'Fitness per different fitness types [goal={goal}]')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'boxplot_{goal}{FIG_SUF}'), format='pdf', facecolor=(1,1,1,0))
    plt.clf()

    # # Per-goal per-fitness_type boxplots 
    # names = list(df_long_goal['strategy'].unique())
    # for grp, name_list in split_fitness(list(df_long_goal['strategy'].unique())):
    #     df_long_goal_names = df_long_goal[df_long_goal['strategy'].isin(name_list)]
    #     grid = sns.catplot(x='strategy', y="fitness", kind="box", data=df_long_goal_names)
    #     for ax in grid.axes.flat:
    #         ax.set_xticklabels(ax.get_xticklabels(), rotation=40, fontweight='bold')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(FIG_DIR, f'boxplot_{goal}_{grp}{FIG_SUF}'), format='pdf')
    #     plt.clf()

    # for idx, runs in enumerate(pair_runs(list(df_boxplot['strategy'].unique()))):
    #     # Fitness lineplot
    #     # Select only the paired runs
    #     df_sub = df_boxplot.loc[df_boxplot['strategy'].isin(runs)]
    #     # Sample the results by a factor of SAMPLE
    #     df_sub = df_sub[df_sub['generation'] % SAMPLE == 0]
    #     grid = sns.relplot(x="generation", y="fitness", hue='strategy', col='fitness_type', kind="line", data=df_sub)
    #     # Put the legend out of the figure
    #     plt.suptitle(f'Fitness per generation [goal={goal}]')
    #     plt.legend(bbox_to_anchor=(1.25, 0.5), frameon=False)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(FIG_DIR, f'fitness_{goal}_{idx}{FIG_SUF}'), format='pdf', facecolor=(1,1,1,0))
    #     plt.clf()

    #     # Time and data ratios plot
    #     grid = sns.relplot(x='generation', y='ratio_values', hue='strategy', col='fitness_type', row='ratio_type', kind='line', data=df_sub)
    #     for ax in grid.axes.flatten():
    #         ax.axhline(y=goal, ls='--', color='red')
    #     plt.suptitle(f'Time and data ratios compared to "No optimization", [goal={goal}]')
    #     plt.legend(bbox_to_anchor=(1.25, 0.5), frameon=False)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(FIG_DIR, f'ratios_{goal}_{idx}{FIG_SUF}'), format='pdf', facecolor=(1,1,1,0))
    #     plt.clf()