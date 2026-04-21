import pandas as pd
import os
import re

CSV_INPUT  = os.path.join("data", "dataset.csv")
EXCEL_OUTPUT = os.path.join("data", "to_label.xlsx")

def clean(text):
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[@#]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = pd.read_csv(CSV_INPUT)
df["comment_body"] = df["comment_body"].astype(str).apply(clean)
df = df[df["comment_body"].str.len() > 20].reset_index(drop=True)
df = df[["comment_id", "comment_body"]].head(500)
df["manual_label"] = ""  # عمود فاضي للتصنيف

with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Comments", index=False)

print(f"Done! File saved: {EXCEL_OUTPUT}")
print("Each person fills the 'manual_label' column with: positive / neutral / negative")
