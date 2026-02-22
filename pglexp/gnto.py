import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from pglexp.base import BaseTrainer, BaseInference

from models.GNTO.models.DataPreprocessor import (
    DataPreprocessor,
    plan_trees_to_graphs,
)
from models.GNTO.models.TrainAndEval import (
    build_dataset,
    train_epoch,
    validate_epoch,
)
from models.GNTO.models.NodeEncoder import NodeEncoder_V4
from models.GNTO.models.TreeEncoder import GATv2TreeEncoder_V3
from models.GNTO.models.PredictionHead import PredictionHead_V2


class GNTOTrainer(BaseTrainer):
    def train(self, input_file: str, epochs: int, batch_size: int, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 1. Collect Data
        raw_data = self.collect_data(input_file)

        if not raw_data:
            print(f"No data found in {input_file}.")
            return

        # Process raw data to extract plans and times
        raw_plans = []
        execution_times = []

        for item in raw_data:
            plan_data = item
            if isinstance(item, (list, tuple)) and len(item) == 2:
                plan_data = item[1]

            if "Plan" in plan_data:
                raw_plans.append(plan_data["Plan"])
                t = plan_data.get("Execution Time", 0.0)
                if t == 0.0:
                    t = plan_data.get("Plan", {}).get("Actual Total Time", 0.0)
                execution_times.append(float(t))

        print(f"Collected {len(raw_plans)} plans.")

        if not raw_plans:
            print("No valid plans found.")
            return

        # 2. Preprocess
        preprocessor = DataPreprocessor()
        print("Converting JSON to PlanNode trees...")
        plan_roots = [preprocessor.preprocess(p) for p in raw_plans]

        # 3. Initialize Metadata & Model
        print("Initializing metadata and model...")
        node_types = set()
        for plan in plan_roots:
            stack = [plan]
            while stack:
                curr = stack.pop()
                node_types.add(curr.node_type)
                stack.extend(curr.children)
        node_types = sorted(list(node_types))
        node_type_map = {t: i for i, t in enumerate(node_types)}

        col_map = {"UNKNOWN": 0}
        op_map = {"=": 0, ">": 1, ">=": 2, "<": 3, "<=": 4, "!=": 5, "<>": 5}

        # Normalization factor
        times_log = [np.log1p(t) for t in execution_times]
        max_time_log = 1.0
        if times_log:
            max_time_log = max(times_log)

        # Define Model Class locally or helper
        class GNTOModel(nn.Module):
            def __init__(self, num_types, num_cols, num_ops):
                super().__init__()
                self.enc = NodeEncoder_V4(num_types, num_cols, num_ops, 16, 2, 64, 64)
                self.tree = GATv2TreeEncoder_V3(64, 64, 64, 4, 2, 0.1)
                self.head = PredictionHead_V2(64, 1, (64, 64))

            def forward(self, d):
                return torch.sigmoid(
                    self.head(self.tree(self.enc(d.x), d.edge_index, d.batch))
                )

        model = GNTOModel(len(node_types), len(col_map) + 100, len(op_map) + 10).to(
            device
        )
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        crit = nn.MSELoss()

        # Save metadata immediately
        save_dir = "models"
        os.makedirs(save_dir, exist_ok=True)
        meta_path = os.path.join(save_dir, "GNTO_meta.json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "node_types": node_types,
                    "max_time_log": float(max_time_log),
                    "node_type_map": node_type_map,
                    "col_map": col_map,
                    "op_map": op_map,
                },
                f,
                indent=2,
            )
        print(f"Metadata saved to {meta_path}")

        # 4. Build Graphs & Features
        print("Building graphs and features...")
        edges_list, _ = plan_trees_to_graphs(plan_roots)
        processed_plans = []
        for root in tqdm(plan_roots, desc="Processing"):
            nodes = []

            def dfs(node):
                nodes.append(
                    {
                        "node_type_id": node_type_map.get(node.node_type, 0),
                        "plan_rows": np.log1p(
                            float(node.extra_info.get("Plan Rows", 0))
                        ),
                        "plan_width": np.log1p(
                            float(node.extra_info.get("Plan Width", 0))
                        ),
                        "predicate_list_processed": [],
                    }
                )
                for c in node.children:
                    dfs(c)

            dfs(root)
            processed_plans.append(nodes)

        # 5. Build Dataset
        print("Building PyG dataset...")
        times_norm = [t / max_time_log for t in times_log]
        ds = build_dataset(
            processed_plans, edges_list, times_norm, in_dim=15, max_len=3
        )

        # Split (simple 80/20 split)
        split = int(len(ds) * 0.8)
        if split == 0:
            split = len(ds)
        train_ds = ds[:split]
        val_ds = ds[split:] if split < len(ds) else None

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds else None

        # 6. Train Loop
        print("Starting training...")
        for e in range(epochs):
            loss = train_epoch(model, train_loader, opt, crit, device)
            val_msg = ""
            if val_loader:
                val_loss, _, _ = validate_epoch(model, val_loader, crit, device)
                val_msg = f"| Val Loss: {val_loss:.4f}"
            print(f"Epoch {e + 1}/{epochs} | Loss: {loss:.4f} {val_msg}")

        # Final Save
        save_path = os.path.join(save_dir, "GNTO_trained.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Final model saved to {save_path}")


class GNTOInference(BaseInference):
    def __init__(self):
        self.model = None
        self.metadata = None
        self.device = None

    def load(self, model_dir: str, device: str = "cpu"):
        self.device = torch.device(device)
        model_name = "GNTO"

        model_path = os.path.join(model_dir, f"{model_name}_trained.pth")
        meta_path = os.path.join(model_dir, f"{model_name}_meta.json")

        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Model artifacts for {model_name} not found in {model_dir}"
            )

        print(f"Loading metadata from {meta_path}...")
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        self.DataPreprocessor = DataPreprocessor
        self.plan_trees_to_graphs = plan_trees_to_graphs
        self.build_dataset = build_dataset

        num_types = len(self.metadata["node_types"])
        num_cols = len(self.metadata["col_map"])
        num_ops = len(self.metadata["op_map"])

        class GNTOModel(nn.Module):
            def __init__(self, num_types, num_cols, num_ops):
                super().__init__()
                self.enc = NodeEncoder_V4(num_types, num_cols, num_ops, 16, 2, 64, 64)
                self.tree = GATv2TreeEncoder_V3(64, 64, 64, 4, 2, 0.1)
                self.head = PredictionHead_V2(64, 1, (64, 64))

            def forward(self, d):
                return torch.sigmoid(
                    self.head(self.tree(self.enc(d.x), d.edge_index, d.batch))
                )

        print(f"Loading model from {model_path}...")
        self.model = GNTOModel(num_types, num_cols + 100, num_ops + 10).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, plans: list) -> list[float]:
        # Preprocessing
        preprocessor = self.DataPreprocessor()
        plan_roots = [preprocessor.preprocess(p) for p in plans]
        edges_list, _ = self.plan_trees_to_graphs(plan_roots)

        processed_plans = []
        node_type_map = self.metadata["node_type_map"]

        for root in plan_roots:
            nodes = []

            def dfs(node):
                nodes.append(
                    {
                        "node_type_id": node_type_map.get(node.node_type, 0),
                        "plan_rows": np.log1p(
                            float(node.extra_info.get("Plan Rows", 0))
                        ),
                        "plan_width": np.log1p(
                            float(node.extra_info.get("Plan Width", 0))
                        ),
                        "predicate_list_processed": [],
                    }
                )
                for c in node.children:
                    dfs(c)

            dfs(root)
            processed_plans.append(nodes)

        times = [0.0] * len(processed_plans)
        dataset = self.build_dataset(
            processed_plans, edges_list, times, in_dim=15, max_len=3
        )
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        all_preds = []
        max_time_log = self.metadata["max_time_log"]

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch).view(-1)

                # Denormalize
                preds_norm = out.cpu().numpy()
                preds_log = preds_norm * max_time_log
                preds = np.expm1(preds_log)
                all_preds.extend(preds.tolist())

        return all_preds
