import pandas as pd
import json
import os
import sys
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EXCEL_INPUT   = os.path.join("data", "after_label.xlsx")
RESULTS_FILE  = os.path.join("data", "gpt_labels.csv")
BATCH_SIZE    = 30

# ── 1. Load & clean labels ──────────────────────────────────────
df = pd.read_excel(EXCEL_INPUT)
df["manual_label"] = df["manual_label"].astype(str).str.strip().str.lower()
df["manual_label"] = df["manual_label"].replace({"nagative": "negative"})
df = df[df["manual_label"].isin(["positive", "neutral", "negative"])].reset_index(drop=True)
print(f"Loaded {len(df)} valid labeled comments.")

# ── 2. Get GPT labels (resume if already started) ───────────────
if os.path.exists(RESULTS_FILE):
    done_df = pd.read_csv(RESULTS_FILE)
    done_ids = set(done_df["comment_id"].astype(str))
    print(f"Resuming: {len(done_df)} already processed.")
else:
    done_df = pd.DataFrame(columns=["comment_id", "gpt_label"])
    done_ids = set()

remaining = df[~df["comment_id"].astype(str).isin(done_ids)]
new_rows = []

for i in range(0, len(remaining), BATCH_SIZE):
    batch = remaining.iloc[i:i + BATCH_SIZE]
    comments_text = "\n".join(f"{j+1}. {c[:300]}" for j, c in enumerate(batch["comment_body"]))

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis expert. "
                        "Label each comment as exactly one of: positive, neutral, or negative. "
                        "Return ONLY valid JSON: {\"labels\": [\"positive\", \"neutral\", ...]}. "
                        "Array length must equal number of input comments."
                    )
                },
                {
                    "role": "user",
                    "content": f"Label these {len(batch)} comments:\n\n{comments_text}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        labels = json.loads(response.choices[0].message.content).get("labels", [])
        while len(labels) < len(batch):
            labels.append("neutral")
        labels = labels[:len(batch)]

        for cid, lbl in zip(batch["comment_id"], labels):
            new_rows.append({"comment_id": cid, "gpt_label": lbl.strip().lower()})

        done_count = len(done_df) + len(new_rows)
        print(f"  Processed {done_count}/{len(df)}")

    except Exception as e:
        print(f"  Error on batch {i}: {e}")
        break

if new_rows:
    done_df = pd.concat([done_df, pd.DataFrame(new_rows)], ignore_index=True)
    done_df.to_csv(RESULTS_FILE, index=False)

# ── 3. Compute metrics ──────────────────────────────────────────
merged = df.merge(done_df, on="comment_id")
merged = merged[merged["gpt_label"].isin(["positive", "neutral", "negative"])]

LABELS = ["positive", "neutral", "negative"]

correct = (merged["manual_label"] == merged["gpt_label"]).sum()
accuracy = correct / len(merged) * 100

print(f"\n{'='*50}")
print(f"  EVALUATION RESULTS ({len(merged)} comments)")
print(f"{'='*50}")
print(f"  Accuracy:  {accuracy:.1f}%")
print()

for label in LABELS:
    tp = ((merged["manual_label"] == label) & (merged["gpt_label"] == label)).sum()
    fp = ((merged["manual_label"] != label) & (merged["gpt_label"] == label)).sum()
    fn = ((merged["manual_label"] == label) & (merged["gpt_label"] != label)).sum()

    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  {label.capitalize():<10} Precision: {precision:.1f}%  Recall: {recall:.1f}%  F1: {f1:.1f}%")

print()
print("  Confusion Matrix (rows=manual, cols=GPT):")
print(f"  {'':12} {'Positive':>10} {'Neutral':>10} {'Negative':>10}")
for actual in LABELS:
    row = [((merged['manual_label']==actual) & (merged['gpt_label']==pred)).sum() for pred in LABELS]
    print(f"  {actual.capitalize():<12} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

print(f"\n  Results saved to: {RESULTS_FILE}")
