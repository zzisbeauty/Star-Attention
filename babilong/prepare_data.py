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

import datasets

from prompts import DEFAULT_PROMPTS, get_formatted_input
from template import PROMPT_TEMPLATES

TASKS = ['qa1', 'qa2', 'qa3', 'qa4', 'qa5']
SPLIT_NAMES = ['16k', '32k', '64k', '128k']


def load_data(dataset_path, split_name, task):
    if not os.path.exists(os.path.join(dataset_path, split_name)):
        os.makedirs(dataset_path, exist_ok=True)
        dataset_name = 'RMT-team/babilong-1k-samples' if int(split_name[:-1]) < 64 else 'RMT-team/babilong'
        data = datasets.load_dataset(dataset_name, split_name)
        data.save_to_disk(os.path.join(dataset_path, split_name))  # Save for future loading
        task_data = data[task]
    else:
        task_data = datasets.load_from_disk(os.path.join(dataset_path, split_name))[task]  # type: ignore

    return task_data


def main(download_dir, output_dir, tasks, split_names, model_template_type):
    os.makedirs(output_dir, exist_ok=True)
    print('Preparing data...')
    for task in tasks:
        for split_name in split_names:
            output_file = os.path.join(output_dir, f'{task}_{split_name}.jsonl')
            if os.path.exists(output_file):
                print(f'{output_file} already exists. Skipping...')
                continue

            task_data = load_data(download_dir, split_name, task)
            with open(output_file, 'w') as f:
                for sample_idx in range(len(task_data)):
                    input_context, input_query = get_formatted_input(
                        task_data[sample_idx]['input'],
                        task_data[sample_idx]['question'],
                        DEFAULT_PROMPTS[task]['examples'],
                        DEFAULT_PROMPTS[task]['instruction'],
                        DEFAULT_PROMPTS[task]['post_prompt'],
                        chat_template=PROMPT_TEMPLATES[model_template_type]['template'],
                    )
                    f.write(
                        json.dumps(
                            {
                                'index': sample_idx,
                                'input_context': input_context,
                                'input_query': input_query,
                                'outputs': [task_data[sample_idx]['target']],
                            }
                        )
                        + '\n'
                    )

            print(f'Data generated: {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_dir', required=True, help='path to download dataset')
    parser.add_argument('--output_dir', required=True, help='path to save the formatted dataset')
    parser.add_argument('--tasks', nargs='+', default=TASKS, choices=TASKS, help='tasks to format')
    parser.add_argument(
        '--split_names',
        nargs='+',
        default=SPLIT_NAMES,
        choices=SPLIT_NAMES,
        help='splits to format for the given task',
    )
    parser.add_argument(
        '--model_template_type', required=True, choices=PROMPT_TEMPLATES.keys(), help='options in `template.py`'
    )
    args = parser.parse_args()

    main(args.download_dir, args.output_dir, args.tasks, args.split_names, args.model_template_type)
