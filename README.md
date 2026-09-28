# Diagnosing Counterfactual Behavior through Local Representation Geometry

Counterfactual explanations seek small, semantically meaningful changes to an input that alter a model's prediction, helping interpret and audit machine learning systems. In modern vision, language, and multimodal systems, pretrained encoders map inputs to representation spaces, where task-specific classifier heads impose decision boundaries. Crossing a boundary changes the prediction, but the resulting representation may lie far from observed target-class examples. In applications such as biomedical research, a prediction change can mislead if it pushes the representation far from observed alternatives. To address this gap, this work uses local representation geometry to improve counterfactual search and establish a model-level diagnostic framework. A standardized local search probe characterizes each encoder–classifier pair by its counterfactual success, minimum displacement, search effort, and endpoint support. Across multiple domains and widely-used pretrained model families in vision, medical imaging, text, multimodal data, and single-cell biology, models with comparable predictive performance exhibit different counterfactual outcomes. Our framework associates these differences with classifier boundary proximity and local representation geometry, quantified by target-class support and local curvature. Incorporating support into the search objective and adapting step sizes to curvature increases counterfactual success under matched constraints. In a real-world biology case study, the generated counterfactuals align with experimentally observed differences between low- and high-immune response in single cells. Together, these findings establish counterfactual behavior as a distinct dimension beyond predictive performance and provide a framework for model diagnosis and more reliable counterfactual search.

![Overview](docs/overview.png)
Counterfactual behavior is shaped by boundary proximity and local support: given a representation, the code measures whether prediction-changing perturbations are reachable, how far they must move, and whether they terminate in target regions supported by nearby data.

---

## News

The repo is currently undergoing construction. Stay tuned for more updates!

---

## Features
- Train linear probes on top of frozen image, text, and multimodal encoders.
- Run standardized local counterfactual search in representation space.
- Measure boundary proximity, local support, counterfactual success, distance, and optimization effort.
- Retrain classifier heads on fixed embeddings to isolate boundary-placement effects.
- Cache embeddings for reproducible and efficient re-use across experiments.
- Generate paper-style figures and case-study payloads from saved experiment outputs.

---

## Supported Datasets & Encoders
- `mnist`, `shapes`, `chestxray`, `imdb`, `mmimdb`
- Vision: `resnet50`, `vit`, `dinov2`
- Text: `distilbert`, `bert`, `roberta`
- Multimodal: `clip`, `siglip2`

---

## Installation
- Requires Python `3.10+`

### Create and activate a virtual environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### Install dependencies
The requirements pin the versions used for the local checks. Full dataset experiments have not been rerun with this environment.

```bash
pip install -r requirements.txt
```

### Optional environment variables
The code can also be configured through environment variables:

- `GEOMETRY_DATA_ROOT`: root directory for datasets
- `GEOMETRY_EMBEDDING_CACHE_DIR`: location for cached embeddings
- `GEOMETRY_HF_CACHE`: Hugging Face cache directory

If these are not set, defaults are taken from `src/core/utils.py`.

---

## Repository Layout
```text
geometry/
├── src/
│   ├── core/              # datasets, encoders, classifiers, geometry utilities
│   ├── counterfactuals/   # search, configuration, evaluation
│   ├── experiments/       # experiment runners
│   ├── analysis/          # post hoc analyses
│   └── figures/           # figure and case-study utilities
├── docs/
├── checkpoints/
├── outputs/
└── README.md
```

---

## MNIST Example Workflow
The MNIST pipeline is the simplest end-to-end example in this repository.

### Step 1: Train and evaluate one encoder
The command below runs the `mnist` + `vit` experiment, evaluates counterfactual behavior on the test split, saves cached embeddings, and stores the trained probe.

```bash
python -m src.experiments.unimodal_encoder_comparison \
  --dataset mnist \
  --encoder vit \
  --seed 42 \
  --probe-epochs 100 \
  --probe-lr 1e-3 \
  --probe-weight-decay 1e-4 \
  --eval-split test \
  --reference-split val \
  --k 20 \
  --step-size 1e-2 \
  --max-steps 300 \
  --trust-radius 1.0 \
  --save-probe-dir outputs/checkpoints \
  --output outputs/mnist_vit_encoder_comparison.json
```

This produces:
- cached embeddings for `train`, `val`, and `test`
- a probe checkpoint in `outputs/checkpoints/`
- an experiment JSON file in `outputs/`

### Step 2: Vary the classifier head under fixed embeddings
This reproduces the boundary-variation setting where the encoder stays fixed and only the linear head is retrained.

```bash
python -m src.experiments.unimodal_head_variation \
  --dataset mnist \
  --encoder vit \
  --probe-checkpoint outputs/checkpoints/mnist_vit_seed42_probe.pt \
  --seed 42 \
  --eval-split test \
  --reference-split val \
  --k 20 \
  --step-size 1e-2 \
  --max-steps 300 \
  --trust-radius 1.0 \
  --intervention-seed 43 \
  --intervention-seed 44 \
  --intervention-seed 45 \
  --intervention-seed 46 \
  --intervention-probe-weight-decay 1e-5 \
  --intervention-probe-weight-decay 1e-4 \
  --intervention-probe-weight-decay 1e-3 \
  --output outputs/interventions/mnist_vit_classifier_head_variation.json
```

This writes a JSON payload with:
- the baseline checkpoint result
- multiple retrained probe variants
- per-variant accuracy and counterfactual metrics
- raw example-level outputs

### Step 3: Run post hoc analysis
Run the combined analysis entry point:

```bash
python -m src.analysis.main \
  --compare-dir outputs \
  --cache-dir outputs/cache/embeddings \
  --interventions-dir outputs/interventions \
  --output-dir outputs/analysis \
  --test-fraction 0.2 \
  --split-seed 0 \
  --k 20 \
  --eval-split test \
  --svm-c 1.0
```

Or run individual analysis tasks:

- Geometry prediction
```bash
python -m src.analysis.main \
  --compare-dir outputs \
  --cache-dir outputs/cache/embeddings \
  --interventions-dir outputs/interventions \
  --output-dir outputs/analysis \
  --test-fraction 0.2 \
  --split-seed 0 \
  --task geometry
```

- Supported flips
```bash
python -m src.analysis.main \
  --compare-dir outputs \
  --cache-dir outputs/cache/embeddings \
  --interventions-dir outputs/interventions \
  --output-dir outputs/analysis \
  --k 20 \
  --task supported_flips
```

- SVM probe comparison
```bash
python -m src.analysis.main \
  --compare-dir outputs \
  --cache-dir outputs/cache/embeddings \
  --interventions-dir outputs/interventions \
  --output-dir outputs/analysis \
  --eval-split test \
  --svm-c 1.0 \
  --task svm_probe
```


### Support sensitivity

Run the paper's neighborhood-size and support-quantile sweeps together:

```bash
python -m src.analysis.main --task support_sensitivity \
  --compare-dir outputs --cache-dir outputs/cache/embeddings \
  --output-dir outputs/analysis \
  --ks 5 10 20 50 --quantiles 0.5 0.75 0.9
```

`density_robustness.json` reports support/success correlations and held-out R²
for each k, including scalar-normalized support. `support_threshold_sensitivity.json`
reports supported success for each quantile and model-ranking agreement with p=0.5,
using each run's recorded k (20 for the main experiments). Thresholds come only
from leave-one-out reference radii; searches and evaluated examples remain fixed.
These are sensitivity analyses, not selection of k or p using test outcomes.
The separate `experiments.model_specific_tuning` runner selects step size and
locality radius using validation data, as described in the paper.


### Papalexi embedding extraction

Install AnnData and the relevant model package in an extraction environment using
[scGPT's installation instructions](https://github.com/bowang-lab/scGPT) or
[Geneformer's installation instructions](https://huggingface.co/ctheodoris/Geneformer).
These optional packages are not imported by the other experiments. Their upstream
dependencies may require separate environments; the main requirements file does
not claim to specify a tested environment for them.

Supply a quality-controlled `.h5ad` containing **all measured genes and raw counts**.
The runner does not infer immune-response labels or create a new experimental split.
The split column must contain `train`, `val`, and `test`, and may also contain
`observed` for independent observed-response cells. Cell IDs must be unique.
Provide gene symbols for scGPT and Ensembl IDs for Geneformer. The column names and
checkpoint paths below are examples; replace them with the actual paper inputs.
Run extraction in a compute job, not on the login node.

```bash
python -m src.experiments.extract_single_cell \
  --input data/papalexi.h5ad --counts-layer counts \
  --encoder scgpt --model-dir checkpoints/scgpt \
  --label-col response --low-label low --high-label high \
  --split-col split --group-cols sample perturbation \
  --gene-name-col gene_symbol --reference-split val \
  --output-dir outputs/cache/papalexi

python -m src.experiments.extract_single_cell \
  --input data/papalexi.h5ad --counts-layer counts \
  --encoder geneformer --model-dir checkpoints/geneformer \
  --geneformer-version V1 --ensembl-col ensembl_id \
  --label-col response --low-label low --high-label high \
  --split-col split --group-cols sample perturbation \
  --gene-name-col gene_symbol --reference-split val \
  --output-dir outputs/cache/papalexi
```

For scGPT, the checkpoint directory contains `args.json`, `vocab.json`, and
`best_model.pt`. Extraction uses the official binned-expression CLS embedding API
with its output normalization and a default token limit of 1,200. For Geneformer,
use a local pretrained model directory and the matching official `V1` or `V2`
model family. The official tokenizer uses that family's gene medians and token
vocabulary. Extraction uses mean-pooled cell embeddings, no cell subsampling,
and the penultimate layer by default (`--emb-layer 0` selects the final layer).
The upstream Geneformer extractor requires CUDA. These extraction defaults must
be checked against the checkpoints and settings used in the paper.

Outputs are `outputs/cache/papalexi/{scgpt,geneformer}/{train,val,test}.pt`, plus
`observed.pt` when that split is present. Each includes cell IDs, binary labels
(low=0, high=1), sample/perturbation groups, embeddings, and aligned expression.
Geneformer results are reordered by cell ID after extraction. Decoder expression
is library-size normalized to 10,000 and log1p transformed. The 4,000 most variable
genes are selected using only the requested reference split; both encoders still
receive all measured genes. Every output records these settings and package versions.

Pass the three `.pt` files to `experiments.cached_probe` with `--dataset papalexi`
and the appropriate `--encoder`. For `experiments.single_cell`, use scGPT's
`val.pt` for `--reference` and `--validation`, `test.pt` for `--test`, and the
saved probe checkpoint. Use the declared observed split for `--observed`, or the
training split if it contains the paper's matched observed-response groups.
The observed group column definitions must match those of the starting cells.
