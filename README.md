# function-understanding

This project turns C source code into function-level semantic artifacts. It parses each source file, extracts individual functions, builds AST-based features, asks an LLM for a high-level purpose description, and saves the resulting datapoints as JSONL. Those datapoints are then embedded, clustered by semantic purpose, and used for retrieval so you can query similar functions from a saved embedding corpus.

The current demo is built around the `rcutils` repository, which is already checked in as a submodule-style C codebase under this workspace. I chose it because it contains roughly 30 C source files, which was enough structure to demonstrate extraction, embedding, clustering, and retrieval without making the project feel toy-sized. You can point the same pipeline at another C Git repository by adding it as a submodule under `rcutils/` and re-running the scripts below.

## Repository Overview

```text
function-understanding/
├── README.md : Project guide, setup steps, demo commands, and evaluation notes. You're reading this rn :))
├── src/
│   ├── extract_function_datapoints.py - Parses C files and extracts per-function datapoints with purpose labels.
│   ├── generate_function_embeddings.py - Converts extracted datapoints into embedding JSONL files.
│   ├── clustering_demo.py - Groups embedded functions by semantic purpose.
│   ├── retrieval_demo.py - Retrieves similar functions for a chosen query function.
│   ├── evaluate_embeddings.py - Computes retrieval metrics and clustering sanity checks.
│   └── embedding_demo_utils.py - Shared helpers for loading, similarity, clustering, and payload formatting.
├── data/
│   └── Generated embeddings, demo outputs, and evaluation reports.
├── rcutils/
│   └── Current C source used as the extraction baseline; can be changed to any other repo
├── docs/
│   ├── EMBEDDING_MODEL.md - answer to why I chose the jina embedding model
│   └── LABEL.md - answer to why I chose the current label extraction method
└── config/
    ├── label_rules.json - Dictionary with AST labelling features and side effect patterns
    └── system_prompt.txt - LLM System prompt for generating high_level_purpose label
```

## Using A C Repository

To swap in another C repository, add it under `rcutils/` as a Git submodule and treat that directory as the source root for extraction. The rest of the pipeline stays the same: the extractor walks the C files, the embedding step consumes the generated JSONL, and the demo/evaluation scripts operate on the saved outputs in `data/`.

For example, from the repository root you can add a new C repository as a submodule like this:

```bash
git submodule add <repository-url> rcutils/<repository-name>
git submodule update --init --recursive
```

In the current workspace, the existing `rcutils` tree provides a realistic baseline with enough function variety to make the semantic outputs useful for a demo.

## Workflow

1. Extract function-level datapoints into JSONL.
2. Generate embeddings for the saved datapoints.
3. Run semantic clustering and retrieval demos.
4. Evaluate the saved demo outputs.

## Extract Functions

Run the extractor from the repository root:

```bash
python3 src/extract_function_datapoints.py --output data/rcutils_function_datapoints.jsonl
```

This script parses the configured C source tree, computes AST-based features, generates an LLM-backed high-level purpose label, and writes the function datapoints to a JSONL file in `data/`.

## Generate embeddings

Run this from the repository root after extraction:

```bash
python3 src/generate_function_embeddings.py --data-dir data --overwrite
```

The script reads the JSONL function datapoints and writes embedding files next to the inputs, for example:

- `data/rcutils_function_datapoints_gpt120b_embeddings.jsonl`
- `data/rcutils_function_datapoints_kimi_embeddings.jsonl`

## Demo Semantic Use Cases

Run the focused demo scripts against the saved embedding files:

```bash
python3 src/retrieval_demo.py --query-function __default_allocate --top-k 5
python3 src/clustering_demo.py --clusters 5
```

If you prefer positional selection instead of a function name, use `--query-index 0` instead of `--query-function`.

The scripts write these result files into `data/`:

- `function_retrieval.json`
- `semantic_purpose_clustering.json`

## Evaluation methodology

Run the evaluation script to score the saved demo outputs and generate a quantitative + qualitative report:

```bash
python3 src/evaluate_embeddings.py
```

If you want Recall@10 to reflect a full ten-result ranking, regenerate the retrieval demo first with `--top-k 10`.

The evaluation writes these files into `data/`:

- `embedding_evaluation_report.json`
- `embedding_evaluation_report.md`

The report includes:

- retrieval MRR and Recall@K using semantic-purpose label agreement
- strict and soft retrieval variants to separate exact label matches from near-matches
- clustering purity, cluster-size imbalance, empty-cluster checks, and singleton-cluster checks
- concrete success and failure cases for human review

## Lightweight UI

Launch the browser UI from the repository root with:

```bash
python3 app.py
```

The UI uses the existing retrieval and clustering demos under `src/` and exposes two tabs:

- Function Search: live function-name suggestions, retrieval results, and strict/soft query metrics
- Cluster Explorer: a cluster slider, scatter plot, and datapoint navigator with name, file path, label, and code preview

The default cluster count is `5`, which is the current best tradeoff for this corpus because it keeps clusters balanced without creating many tiny or empty groups.
