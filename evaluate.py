import pandas as pd

df_manual = pd.read_excel("data/manual_balanced_1000.xlsx")
df_gpt    = pd.read_excel("data/after_label1.xlsx")

merged = df_manual.merge(df_gpt, on="comment_id", suffixes=("_true", "_pred"))

true_labels = merged["manual_label_true"].str.lower().str.strip().tolist()
pred_labels = merged["manual_label_pred"].str.lower().str.strip().tolist()

print(f"Total comments compared: {len(true_labels)}")

classes = ["positive", "neutral", "negative"]
results = {}
for cls in classes:
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == cls and p != cls)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results[cls] = {"precision": precision, "recall": recall, "f1": f1,
                    "tp": tp, "fp": fp, "fn": fn}

macro_p  = sum(r["precision"] for r in results.values()) / len(classes)
macro_r  = sum(r["recall"]    for r in results.values()) / len(classes)
macro_f1 = sum(r["f1"]        for r in results.values()) / len(classes)
accuracy = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(true_labels)

print("\n" + "="*55)
print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-"*55)
for cls in classes:
    r = results[cls]
    print(f"{cls:<12} {r['precision']:>10.2%} {r['recall']:>10.2%} {r['f1']:>10.2%}")
print("-"*55)
print(f"{'Macro Avg':<12} {macro_p:>10.2%} {macro_r:>10.2%} {macro_f1:>10.2%}")
print("="*55)
print(f"Overall Accuracy: {accuracy:.2%}")
print(f"Total comments:   {len(true_labels)}")
