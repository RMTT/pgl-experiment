from abc import ABC, abstractmethod
import numpy as np
import os
import json


class BaseTrainer(ABC):
    """
    Abstract base class for model trainers.
    """

    def __init__(self):
        """
        Initialize the trainer.
        """
        pass

    @abstractmethod
    def train(self, input_file: str, epochs: int, batch_size: int, **kwargs):
        """
        Train the model using data from the input file.

        Args:
            input_file: Path to a file containing JSONs from qdataset_collect.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
        """
        pass

    def collect_data(self, input_file: str):
        """
        Collect dataset from a file.

        Args:
            input_file: Path to a file containing JSONs from qdataset_collect.

        Returns:
            list: List of data items (dicts) or None on error.
        """
        if not input_file:
            print("Error: input_file must be provided.")
            return None

        print(f"Loading dataset from file: {input_file}...")
        try:
            data_list = []
            with open(input_file, "r") as f:
                # Try to detect if it's a list or objects
                first_char = f.read(1)
                f.seek(0)

                is_list = first_char == "["

                if is_list:
                    data_list = json.load(f)
                else:
                    # JSONL
                    data_list = [json.loads(line) for line in f if line.strip()]

            return data_list

        except Exception as e:
            print(f"Error reading input file: {e}")
            return None


class BaseInference(ABC):
    """
    Abstract base class for model inference.
    """

    @abstractmethod
    def load(self, model_dir: str, device: str = "cpu"):
        """
        Load the trained model and metadata.

        Args:
            model_dir: Directory containing model artifacts.
            device: Device to load the model on ("cpu" or "cuda").
        """
        pass

    @abstractmethod
    def predict(self, plans: list) -> list[float]:
        """
        Predict costs/metrics for a list of query plans.

        Args:
            plans: List of plan dictionaries (JSON-like).

        Returns:
            List of predicted float values (e.g., estimated execution time or cost).
        """
        pass


# Try to import PglAdapter, define dummy if missing (for dev environments without pgl)
try:
    from pgl import PglAdapter
except ImportError:

    class PglAdapter:
        pass


class ModelServingAdapter(PglAdapter):
    """
    Generic adapter that wraps a BaseInference implementation for the pglearned server.
    """

    def __init__(self, inference_engine: BaseInference):
        self.inference = inference_engine

    def choose_plan(self, plans: list, query: str = None) -> int:
        """
        Selects the best plan from a list of candidates.

        Args:
            plans: List of candidate query plans.
            query: The SQL query string (optional).

        Returns:
            Index of the best plan (lowest predicted cost).
        """
        if not plans:
            return 0

        # Extract 'Plan' part if wrapped (pglearned sometimes wraps in 'Plan' key, sometimes not)
        raw_plans = []
        for p in plans:
            if isinstance(p, dict) and "Plan" in p:
                raw_plans.append(p["Plan"])
            else:
                raw_plans.append(p)

        costs = self.inference.predict(raw_plans)

        # Find index of minimum cost
        best_idx = int(np.argmin(costs))
        print(f"Predicted costs: {costs}. Choosing plan {best_idx}.")
        return best_idx
