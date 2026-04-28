"""
Flowers Type — Decision Tree Classifier
---------------------------------------
Trains a Decision Tree on a small flower dataset
(sepal_length, petal_length -> class) and reports test-set accuracy.

Run with defaults:
    python Flowers_Type.py

Run with options:
    python Flowers_Type.py --csv flower.csv --max-depth 4 --test-size 0.30 --seed 7 --save-plot tree.png
"""

import argparse
import sys

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def parse_args():
    p = argparse.ArgumentParser(description="Train a Decision Tree on the flower dataset.")
    p.add_argument("--csv", default="flower.csv", help="Path to the CSV dataset (default: flower.csv)")
    p.add_argument("--max-depth", type=int, default=3, help="Max depth of the decision tree (default: 3)")
    p.add_argument("--test-size", type=float, default=0.30, help="Fraction of data for testing (default: 0.30)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible splits (default: None)")
    p.add_argument("--save-plot", default=None, help="If set, save a tree visualization to this PNG path")
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Load data
    flower_data = pd.read_csv(args.csv)
    print("The dataset:\n\n", flower_data, "\n")

    # 2) Split features and label
    features = flower_data[["sepal_length", "petal_length"]]
    label = flower_data[["class"]]

    # 3) Train / test split
    x_train, x_test, y_train, y_test = train_test_split(
        features, label, test_size=args.test_size, random_state=args.seed
    )

    # 4) Train the model
    model = DecisionTreeClassifier(max_depth=args.max_depth, random_state=args.seed)
    model.fit(x_train, y_train)

    # 5) Predict
    prediction = model.predict(x_test)
    print("\nThe actual flower type:\n", y_test["class"].to_numpy())
    print("\nThe predicted flower type:\n", prediction)

    # 6) Evaluate
    acc = accuracy_score(y_test, prediction)
    print(f"\nMy decision tree accuracy = {acc:.4f}")

    # 7) Optional tree visualization
    if args.save_plot:
        try:
            import matplotlib.pyplot as plt
            from sklearn.tree import plot_tree
        except ImportError:
            print("\n[!] matplotlib is not installed. Run: pip install matplotlib", file=sys.stderr)
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_tree(
            model,
            feature_names=["sepal_length", "petal_length"],
            class_names=sorted(label["class"].unique()),
            filled=True,
            ax=ax,
        )
        fig.tight_layout()
        fig.savefig(args.save_plot, dpi=150)
        print(f"\nTree visualization saved to: {args.save_plot}")


if __name__ == "__main__":
    main()
