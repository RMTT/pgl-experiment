# pgl-experiment

A framework for training and serving machine learning models for query plan execution time estimation, particularly designed to integrate with the `pglearned` ecosystem. 

This repository allows you to take PostgreSQL query plan datasets, train PyTorch-based neural networks (like the `GNTO` model), and run a gRPC inference server to provide live execution time predictions.

## Project Structure

- `run.py`: The main entry point CLI for both training and serving models.
- `pglexp/`: Contains the model adapters and architecture definitions.
  - `base.py`: Defines the `BaseTrainer` and `BaseInferencer` abstract classes that all models must implement.
  - `gnto.py`: The implementation of the GNTO (Graph Neural Tree Optimizer) model, including its trainer and inferencer logic.
- `models/`: List supported models
- `saved_models`: metadata and weights for trained model

## Prerequisites

Ensure you have all necessary Python dependencies installed. Generally, you will need:
- `torch`
- `torch-geometric`
- `numpy`
- `tqdm`
- `pgl` (The `pglearned` python library)

## How to Use

The framework is operated primarily through the `run.py` script.

### 1. Training a Model

To train a model, you need a dataset of PostgreSQL query plans in JSON/JSONL format (typically collected via `qdataset_collect`). 

**Command:**
```bash
python run.py train <model_name> --input <path_to_dataset> [options]
```

**Options:**
- `model`: The name of the model to train (e.g., `GNTO`). This must match a file inside the `pglexp/` directory (e.g., `pglexp/gnto.py`).
- `--input`: Required. The path to your JSON dataset file which can be generate via [pgl cli](https://github.com/RMTT/pglearned/tree/main/cli).
- `--epochs`: (Optional) Number of training epochs. Default is `10`.
- `--batch-size`: (Optional) The batch size for training. Default is `32`.

**Example:**
```bash
python run.py train GNTO --input data/query_plans.json --epochs 20 --batch-size 64
```
*Note: Trained models and their metadata are automatically saved into the `saved_models/` directory by default.*

---

### 2. Serving a Model (Inference)

Once a model is trained, you can spin up a gRPC inference server. This allows tools like `pglearned` to send candidate query plans to this server and receive estimated execution times to assist in query optimization.

**Command:**
```bash
python run.py serve <model_name> [options]
```

**Options:**
- `model`: The name of the model to serve (e.g., `GNTO`).
- `--host`: (Optional) Host address to bind the server to. Default is `0.0.0.0`.
- `--port`: (Optional) Port to bind the server to. Default is `50051`.

**Example:**
```bash
python run.py serve GNTO --port 50051
```
*Note: Ensure your trained model artifacts (`*_trained.pth` and `*_meta.json`) are placed inside the `saved_models/` directory (or wherever your `Inferencer` is configured to load them from) before starting the server.*

## Adding a New Model

To add a new model to the framework:
1. Create a new python file inside the `pglexp/` directory (e.g., `pglexp/my_model.py`).
2. Inside that file, implement two classes:
   - `<ModelName>Trainer` inheriting from `pglexp.base.BaseTrainer`
   - `<ModelName>Inferencer` inheriting from `pglexp.base.BaseInferencer`
3. Call it dynamically via `python run.py train my_model --input ...`
