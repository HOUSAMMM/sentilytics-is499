import pandas as pd
import json
import re
import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TARGETS   = {"positive": 395, "neutral": 287, "negative": 318}
BATCH_SIZE = 30

def clean(text):
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[@#]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = pd.read_csv("data/dataset.csv")
df["comment_body"] = df["comment_body"].astype(str).apply(clean)
df = df[df["comment_body"].str.len() > 20].reset_index(drop=True)

collected = {"positive": [], "neutral": [], "negative": []}
total_collected = lambda: sum(len(v) for v in collected.values())

i = 0
while total_collected() < sum(TARGETS.values()) and i < len(df):
    batch = df.iloc[i:i+BATCH_SIZE]
    i += BATCH_SIZE

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

        for (_, row), lbl in zip(batch.iterrows(), labels[:len(batch)]):
            lbl = lbl.strip().lower()
            if lbl in collected and len(collected[lbl]) < TARGETS[lbl]:
                collected[lbl].append({
                    "comment_id":   row["comment_id"],
                    "comment_body": row["comment_body"],
                    "gpt_label":    lbl
                })

        print(f"  Positive: {len(collected['positive'])}/{TARGETS['positive']}  "
              f"Neutral: {len(collected['neutral'])}/{TARGETS['neutral']}  "
              f"Negative: {len(collected['negative'])}/{TARGETS['negative']}")

        if all(len(collected[k]) >= TARGETS[k] for k in TARGETS):
            break

    except Exception as e:
        print(f"  Error: {e}")

# Combine and shuffle
all_rows = collected["positive"] + collected["neutral"] + collected["negative"]
result = pd.DataFrame(all_rows).sample(frac=1, random_state=42).reset_index(drop=True)

# Save GPT labels
result[["comment_id", "gpt_label"]].to_csv("data/gpt_balanced_1000.csv", index=False)

# Save Excel for manual labeling (without gpt_label)
excel_df = result[["comment_id", "comment_body"]].copy()
excel_df["manual_label"] = ""
excel_df.to_excel("data/manual_balanced_1000.xlsx", index=False)

print(f"\nDone! Total: {len(result)} comments")
print(f"Positive: {len(collected['positive'])}  Neutral: {len(collected['neutral'])}  Negative: {len(collected['negative'])}")
print("Saved: data/manual_balanced_1000.xlsx")
print("Saved: data/gpt_balanced_1000.csv")
