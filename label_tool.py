import pandas as pd
import os
import re

CSV_INPUT  = os.path.join("data", "dataset.csv")
CSV_OUTPUT = os.path.join("data", "labeled_500.csv")
SAMPLE_SIZE = 500

def clean(text):
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[@#]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load dataset
df = pd.read_csv(CSV_INPUT)
df["comment_body"] = df["comment_body"].astype(str).apply(clean)
df = df[df["comment_body"].str.len() > 20].reset_index(drop=True)

# Load progress if exists
if os.path.exists(CSV_OUTPUT):
    done_df = pd.read_csv(CSV_OUTPUT)
    done_ids = set(done_df["comment_id"].astype(str).tolist())
    print(f"Resuming... {len(done_df)} already labeled.")
else:
    done_df = pd.DataFrame(columns=["comment_id", "comment_body", "manual_label"])
    done_ids = set()

# Filter out already labeled
remaining = df[~df["comment_id"].astype(str).isin(done_ids)].head(SAMPLE_SIZE - len(done_df))

if len(remaining) == 0:
    print("All 500 comments labeled!")
else:
    print(f"\n{'='*60}")
    print("  LABELING TOOL — 500 Comments")
    print("  Commands: p=positive  n=negative  neu=neutral  q=quit")
    print(f"{'='*60}\n")

    labeled = []
    total_done = len(done_df)

    for _, row in remaining.iterrows():
        total_done += 1
        print(f"\n[{total_done}/500] -------------------------------------")
        print(f"{row['comment_body'][:300]}")
        print()

        while True:
            choice = input("Label (p / n / neu / q): ").strip().lower()
            if choice == "p":
                label = "positive"
                break
            elif choice == "n":
                label = "negative"
                break
            elif choice == "neu":
                label = "neutral"
                break
            elif choice == "q":
                print("Progress saved. Run again to continue.")
                if labeled:
                    new_df = pd.DataFrame(labeled)
                    done_df = pd.concat([done_df, new_df], ignore_index=True)
                    done_df.to_csv(CSV_OUTPUT, index=False)
                exit()
            else:
                print("  Invalid. Type p, n, neu, or q")

        labeled.append({
            "comment_id":   row["comment_id"],
            "comment_body": row["comment_body"],
            "manual_label": label
        })
        print(f"  OK: {label}")

        # Save every 10 comments
        if len(labeled) % 10 == 0:
            new_df = pd.DataFrame(labeled)
            done_df = pd.concat([done_df, new_df], ignore_index=True)
            done_df.to_csv(CSV_OUTPUT, index=False)
            labeled = []
            print(f"  >> Progress saved ({total_done}/500)")

    # Save remaining
    if labeled:
        new_df = pd.DataFrame(labeled)
        done_df = pd.concat([done_df, new_df], ignore_index=True)
        done_df.to_csv(CSV_OUTPUT, index=False)

    if total_done >= 500:
        print("\nDone! All 500 comments labeled.")
        print(f"Saved to: {CSV_OUTPUT}")
    else:
        print(f"\nProgress saved. {total_done}/500 done.")
