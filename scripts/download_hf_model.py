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
import os

from huggingface_hub import snapshot_download

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', required=True, help='name of the model to download from huggingface hub')
parser.add_argument('-d', '--dir', default=(os.path.join(BASE_DIR, 'hf_models')), help='directory where the model will be downloaded')
parser.add_argument('-t', '--token', default=None, help='huggingface hub token')
args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)
model_dir = os.path.join(args.dir, args.name.replace('/', '_'))

snapshot_download(
    repo_id=args.name,
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    force_download=True,
    cache_dir=os.path.join(args.dir, '.hf_cache'),
    token=args.token,
)
