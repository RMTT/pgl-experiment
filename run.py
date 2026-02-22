import sys
import os
import argparse
import importlib
from pgl.server import run_server

from pglexp.base import ModelServingAdapter


def get_model_module(model_name):
    try:
        return importlib.import_module(f"pglexp.{model_name.lower()}")
    except ImportError as e:
        print(f"Error loading model '{model_name}': {e}")
        print(
            f"Make sure pglexp/{model_name.lower()}.py exists and implements the interface."
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and serve pglearned models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a specific model")
    train_parser.add_argument("model", help="Name of the model to train (e.g., GNTO)")
    train_parser.add_argument(
        "--input",
        dest="input_file",
        required=True,
        help="Input file path containing JSONs from qdataset_collect",
    )
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Serve a trained model")
    serve_parser.add_argument("model", help="Name of the model to serve (e.g., GNTO)")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=50051, help="Port to bind to")

    args = parser.parse_args()

    # Dynamically load the model module
    mod = get_model_module(args.model)

    if args.command == "train":
        # We can try to find a class ending with "Trainer" or "Inferencer" in the module
        trainer_class = getattr(mod, f"{args.model.upper()}Trainer", None)
        if not trainer_class:
            print(f"No Trainer class found in pglexp/{args.model.lower()}.py")
            sys.exit(1)

        print(f"Using trainer: {trainer_class.__name__}")
        trainer = trainer_class()

        trainer.train(
            input_file=args.input_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    elif args.command == "serve":
        inference_class = getattr(mod, f"{args.model.upper()}Inferencer", None)
        if not inference_class:
            print(f"No Inferencer class found in pglexp/{args.model.lower()}.py")
            sys.exit(1)

        print(f"Using inference engine: {inference_class.__name__}")
        inference = inference_class()

        # Load model artifacts
        inference.load("saved_models")

        adapter = ModelServingAdapter(inference)
        run_server(adapter, host=args.host, port=args.port)
