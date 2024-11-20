# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd

from metrics import TASK_LABELS, compare_answers

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASKS = ['qa1', 'qa2', 'qa3', 'qa4', 'qa5']
SPLIT_NAMES = ['16k', '32k', '64k', '128k']


def read_jsonl(file_path):
    with open(file_path) as f:
        return [json.loads(line.strip()) for line in f]


def plot_heatmap(accuracy, title, output_dir):
    import matplotlib
    import matplotlib.pylab as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    # Set large font sizes for better visibility in the PDF
    # matplotlib.rc('font', size=10)

    # Create a colormap for the heatmap
    cmap = LinearSegmentedColormap.from_list('ryg', ["red", "yellow", "green"], N=256)

    # Create the heatmap
    _, ax = plt.subplots(1, 1, figsize=(10, 3.5))  # Adjust the size as necessary
    sns.heatmap(
        accuracy,
        cmap=cmap,
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        xticklabels=SPLIT_NAMES,
        yticklabels=TASKS,
        ax=ax,
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('Context size')
    ax.set_ylabel('Tasks')

    # Save the figure to a PDF
    plt.savefig(f'{output_dir}/{title.lower().replace(" ", "_")}.png', bbox_inches='tight', dpi=300)


def display_accuracy_table(accuracy):
    for idx, seq_len in enumerate(accuracy):
        seq_len_acc = [SPLIT_NAMES[idx]] + [f'{int(score)}' for score in seq_len]
        print(','.join(seq_len_acc))


def display_unfinished_experiments(exp_list, message):
    if exp_list:
        print(f'\n{message}:')
        for task, seq_lengths in exp_list.items():
            print(f'{task}: {" ".join(seq_lengths)}')


def gather_experiment_results(exp_dir, heatmap=False):
    missing_exps, incomplete_exps = defaultdict(list), defaultdict(list)
    accuracy = np.ones((len(SPLIT_NAMES), len(TASKS))) * -1
    for j, task in enumerate(TASKS):
        for i, seq_length in enumerate(SPLIT_NAMES):
            if not os.path.isdir(os.path.join(exp_dir, seq_length)):
                continue

            results_file = os.path.join(exp_dir, seq_length, f'{task}.jsonl')

            if not os.path.isfile(results_file) or os.path.getsize(results_file) == 0:
                if not os.path.isfile(results_file):
                    missing_exps[task].append(seq_length)
                else:
                    incomplete_exps[task].append(seq_length)
                continue

            result_data = read_jsonl(results_file)
            if len(result_data) < 999:
                incomplete_exps[task].append(seq_length)

            df = pd.DataFrame(result_data)
            df['correct'] = df.apply(
                lambda row: compare_answers(
                    target=row['outputs'][0],
                    output=row['pred'],
                    question=(  # FIXME: This is a hack to remove the assistant header - Works only for Llama Instruct Models
                        row['input_query']
                        .replace('<|eot_id|><|start_header_id|>assistant<|end_header_id|>', '')
                        .strip()
                    ),
                    task_labels=TASK_LABELS[task],
                ),
                axis=1,
            )
            score = df['correct'].sum()
            accuracy[i, j] = 100 * score / len(df) if len(df) > 0 else 0

    if heatmap:
        plot_heatmap(accuracy.transpose(), os.path.basename(exp_dir), exp_dir)

    display_accuracy_table(accuracy)

    # Display unfinished experiments
    display_unfinished_experiments(missing_exps, 'Missing Experiments')
    display_unfinished_experiments(incomplete_exps, 'Incomplete Experiments')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e',
        '--exp',
        required=True,
        help=(
            'experiment name containing the results or path to the results directory. '
            'If experiment name is given then it will be searched in `output_dir`. '
            '`exp` is given more priority than `project`.'
        ),
    )
    parser.add_argument(
        '--output_dir',
        default=os.path.join(BASE_DIR, 'results'),
        help='results directory',
    )
    parser.add_argument('-ht', '--heatmap', action='store_true', help='plot heatmap')
    args = parser.parse_args()

    if '/' not in args.exp:
        args.exp = os.path.join(args.output_dir, args.exp)
    if not os.path.isdir(args.exp):
        raise NotADirectoryError(f'Invalid experiment path: {args.exp}')

    gather_experiment_results(args.exp, heatmap=args.heatmap)
