# 🌸 Flowers Type — Decision Tree Classifier

### 🔍 Overview
A simple, beginner-friendly **Decision Tree** classifier that predicts the **flower type** (`Type1`, `Type2`, `Type3`) using two physical measurements: **sepal length** and **petal length**.

Built with `pandas` + `scikit-learn`, this project is designed as a first hands-on experience with supervised learning.

---

### 📦 Features
- Reads the dataset from `flower.csv`
- Splits the data into **train** / **test** sets
- Trains a `DecisionTreeClassifier` with a configurable `max_depth`
- Reports test-set **accuracy**
- **NEW:** All key parameters exposed as **command-line options**
- **NEW:** Optional **tree visualization** saved as a PNG (`--save-plot`)
- **NEW:** Reproducible runs via `--seed`

---

### 🧠 Dataset
| Property | Details |
|-----------|----------|
| **File** | `flower.csv` |
| **Features** | `sepal_length`, `petal_length` |
| **Label** | `class` (Type1, Type2, Type3) |
| **Samples** | ~50 rows |
| **Task** | Multi-class classification |

---

### 🧰 Requirements
```bash
pip install -r requirements.txt
```

---

### ⚙️ How to Run
**Default run**:
```bash
python Flowers_Type.py
```

**Custom run**:
```bash
python Flowers_Type.py --csv flower.csv --max-depth 4 --test-size 0.30 --seed 7 --save-plot tree.png
```

---

### 🛠️ Command-Line Options
| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | `flower.csv` | Path to the CSV dataset |
| `--max-depth` | `3` | Max depth of the decision tree |
| `--test-size` | `0.30` | Fraction of data used for testing (0.0–1.0) |
| `--seed` | `None` | Random seed for reproducible splits |
| `--save-plot` | _off_ | If set, saves a tree visualization to this PNG path |

---

### 📊 Example Output
```
The dataset:
   sepal_length  petal_length  class
0           5.1           1.5  Type1
1           4.8           1.4  Type1
...

The actual flower type:    ['Type2' 'Type1' 'Type3' ...]
The predicted flower type: ['Type2' 'Type1' 'Type3' ...]

My decision tree accuracy = 0.93
```

---

### 🎓 Learning Objectives
1. Read a CSV into a `pandas.DataFrame`
2. Separate **features** from the **label**
3. Use `train_test_split` to create train / test sets
4. Train a `DecisionTreeClassifier` and tune `max_depth`
5. Evaluate the model with `accuracy_score`

---

### 📁 Project Structure
```
Flowers_Type_DT/
├── Flowers_Type.py     # Main training script
├── flower.csv          # Dataset
├── requirements.txt    # Python dependencies
└── README.md
```

---

### 📄 License
Educational use — Credits to **[CS42.org](https://cs42.org)**.
