# Star Attention: Efficient LLM Inference over Long Sequences

This repository contains code for the paper [Star Attention: Efficient LLM Inference over Long Sequences](). Star Attention is a novel block-sparse attention mechanism designed to enable efficient inference on long sequences in transformer-based LLMs. The method operates in two phases:
1. **Phase 1 - Context Encoding**: The context tokens are processed using blockwise-local attention, with the context segmented into blocks where each block is prefixed with an anchor block.
2. **Phase 2 - Query Processing and Token Generation**: The query and response tokens attend to all prior cached tokens through sequence-global attention.

The codebase contains the implementation of Star Attention in PyTorch using the [HuggingFace Transformers](https://github.com/huggingface/transformers) library, along with the code for launching inference with Star Attention on two benchmarks: RULER and BABILong.


## Setup Instructions

### Dependencies

Install all the project dependencies with
```
$ pip install -r requirements.txt
```

In a python shell, download the `punkt` tokenizer from the `nltk` library:
```
import nltk
nltk.download('punkt_tab')
```

### RULER Setup

To generate synthetic data for RULER, you need to download:
- Paul Graham Essays for NIAH from [NIAH Github](https://github.com/gkamradt/LLMTest_NeedleInAHaystack/tree/main/needlehaystack/PaulGrahamEssays) and [Paul Graham Blog](https://paulgraham.com/articles.html).
- QA datasets from [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) and [HotpotQA](https://hotpotqa.github.io/).

To download these data, run:
```
$ bash ruler/download_data.sh
```

### Downloading Models

To download a model from HuggingFace, use the script: [`scripts/download_hf_model.py`](scripts/download_hf_model.py).

*NOTE: For certain models, you might need to input the huggingface hub token from your account settings via the `--token` flag.*

## Launching Inference with Star Attention

This repository contains code for launching inference with Star Attention on two benchmarks: RULER and BABILong. The instructions to run each of those benchmarks are shared in the following subsections.

### RULER
[[Paper](https://arxiv.org/abs/2404.06654) | [GitHub](https://github.com/hsiehjackson/RULER)]

To run inference on RULER, use the script: [`run_ruler.py`](run_ruler.py).

Usage:
```
$ python run_ruler.py \
    -n <experiment_name> \
    -p <path_to_model> \
    -pc <prompt_template_type> \
    -a star \
    -bs <context_block_size> \
    -l <list_of_sequence_lengths_to_run_inference> \
    -np <num_parallel_processes_per_node> \
    --output_dir <output_directory>
```

After running the evaluations, you can display all the results together using the script:
```
$ python ruler/gather_results_ruler.py \
    -e <path_to_results_directory>
```

#### Configuring the `-np` flag

The `-np` flag specifies the number of parallel processes (hosts) to use for running inference. For example, if your machine has 8 GPUs:
- `-np 8`: Launch 8 hosts, each host assigned a single GPU.
- `-np 4`: Lauch 4 hosts, each host assigned 2 GPUs.

This is useful when you want to run star attention with bigger context block sizes or with bigger models where assigning a single GPU per host leads to out-of-memory errors.

To see an example of how to run this script with different configurations, check [`launch.sh`](launch.sh).

#### Configuring the `-nn` flag for Multi-Node Inference

If you have a multi-node setup (such as a slurm cluster), then you can add the `-nn <num_nodes>` for running multi-node inference. The script will launch a total of `nn * np` processes (hosts) for inference.

For example, in a system with 8 GPUs on each node, if `-nn 1` and `-np 4` are specified, the script will launch 4 processes (hosts) on a single node. This means that each host is allocated 2 GPUs to load the model. If the model or the block size is too large, you can scale the `nn` and the `np` parameters accordingly. If `-nn 2` and `-np 2`, then the script will launch a total of 4 processes (hosts) across 2 nodes, with each host containing 4 GPUs.

### BABILong
[[Paper](ttps://arxiv.org/abs/2406.10149) | [GitHub](https://github.com/booydar/babilong)]

To run inference on BABILong, use the script: [`run_babilong.py`](run_babilong.py).

The script takes in the same set of arguments as the RULER script described above. To see an example of how to run this script with different configurations, check [`launch.sh`](launch.sh).

After running the evaluations, you can display all the results together using the script:
```
$ python babilong/gather_results_babilong.py \
    -e <path_to_results_directory>
```

### Running Inference on Custom Data

To run inference on your custom data, use the script: [`run_star_attn_inference.py`](run_star_attn_inference.py). The scripts takes in the input data in `.jsonl` format in which each line of jsonl should look like:
```json
{
  "index": "<sample index>",  # optional
  "input_context": "<long context portion of the input sample>",
  "input_query": "<query portion of the input sample>",
  "output": "<expected output response>",
}
```

Script usage:
```
$ python run_star_attn_inference.py \
    --model_path <path_to_model> \
    --attn_type star \
    --block_size <context_block_size> \
    --tokens_to_generate <num_tokens_to_generate> \
    --stop_words <end_of_sequence_tokens> \
    --input_path <path_to_input_jsonl> \
    --output_path <path_to_output_jsonl>
```

For more details on the script arguments, run:
```
$ python run_star_attn_inference.py --help
```

## Two Phases of Star Attention

Given a system with $H$ hosts and an input sample with context $c$ followed by query $q$, Star Attention operates in two phases:

### Phase 1 - Context Encoding

<div align="center">
  <img
    src="images/star_attn_phase1.png"
    alt="star attention phase 1"
  />
</div>
<br />

- The context is segmented into contiguous blocks:
  $$c = [c_1, c_2, \ldots, c_n]$$
- From the second block, each block $c_i$ is prefixed with $c_1$ - called the anchor block. Thus forming an augmented context:
  $$c' = [c_1, (c_1, c_2), (c_1, c_3), \ldots, (c_1, c_n)]$$
- The augmented context blocks are distributed across the $H$ hosts, with each host attending only to its assigned blocks.
  - After processing the context blocks, each host stores the *non-anchor* portion of the KV cache.

### Phase 2 - Query Processing and Token Generation

<div align="center">
  <img
    src="images/star_attn_phase2.png"
    alt="star attention phase 2"
  />
</div>
<br />

- Designate one host as the *query* host $h_q$.
- Replicate the query tokens to all the hosts where each host first attends to its locally stored KV cache from phase 1.
  $$A_h = \left( \frac{\exp\left( \frac{QK_h^\top}{\sqrt{d}} \right)}{\sum_{k=1}^{l_k} \exp\left( \frac{QK_{h,k}^\top}{\sqrt{d}} \right)} \right)V_h$$
- In addition to the local attention output $A_h$, the hosts also store the sum of exponents from local softmax (i.e. denominator from the equation above).
  $$s_h = \sum_{k=1}^{l_k} \exp\left( \frac{QK_{h,k}^\top}{\sqrt{d}} \right)$$
- The query-host $h_q$ then gathers both the sum of exponents $s_h$ and the local attention output $A_h$ from all hosts:
  $$s = [s_1, s_2, \ldots, s_{H}]$$
  $$A = [A_1, A_2, \ldots, A_{H}]$$
- To compute global attention, the query-host first calculates the global sum of exponents (i.e., the global softmax denominator) as:
  $$s_{\text{global}} = \sum_{h=1}^{H} s_h$$
- Using this global sum, the query-host computes the final global attention output as:
  $$A_{\text{global}} = \sum_{h=1}^{H} \frac{s_h}{s_{\text{global}}} A_h$$

This method ensures that attention scores are correctly normalized across all hosts, requiring only the communication of a single scalar (the sum of exponents, $s_h$) and a vector (the local attention output, $A_h$) per token.


## Citation
```
@article{}
```

## Contact/Getting Help

If you need any help or want to report a bug, feel free to raise an issue in the repo.