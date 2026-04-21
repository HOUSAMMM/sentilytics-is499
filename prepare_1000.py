import pandas as pd
import json
import re
import os
from openai import OpenAI

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

BATCH_SIZE = 30

def clean(text):
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[@#]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load middle 1000
df = pd.read_csv("data/dataset.csv")
df["comment_body"] = df["comment_body"].astype(str).apply(clean)
df = df[df["comment_body"].str.len() > 20].reset_index(drop=True)

mid = len(df) // 2
sample = df.iloc[mid:mid+1000][["comment_id", "comment_body"]].reset_index(drop=True)
print(f"Extracted {len(sample)} comments from the middle.")

# Export Excel for manual labeling
sample["manual_label"] = ""
sample.to_excel("data/manual_1000.xlsx", index=False)
print("Saved: data/manual_1000.xlsx")

# Get GPT labels
gpt_rows = []
for i in range(0, len(sample), BATCH_SIZE):
    batch = sample.iloc[i:i+BATCH_SIZE]
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
        for cid, lbl in zip(batch["comment_id"], labels[:len(batch)]):
            gpt_rows.append({"comment_id": cid, "gpt_label": lbl.strip().lower()})
        print(f"  GPT processed {min(i+BATCH_SIZE, len(sample))}/{len(sample)}")
    except Exception as e:
        print(f"  Error: {e}")

gpt_df = pd.DataFrame(gpt_rows)
gpt_df.to_csv("data/gpt_labels_1000.csv", index=False)
print("Saved: data/gpt_labels_1000.csv")
print("\nDone! Fill manual_label column in manual_1000.xlsx, then run evaluate_1000.py")
