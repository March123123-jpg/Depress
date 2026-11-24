
import pandas as pd
import json

# 1. Load files
with open("labeled_data_DONE/Ploy.xlsx_labeled.json", "r", encoding="utf-8") as f:
    ai = pd.DataFrame(json.load(f))

human = pd.read_csv("Ploy.xlsx.csv")

# 2. Normalize source URLs
ai["source"] = ai["source"].astype(str).str.strip()
human["source"] = human["source"].astype(str).str.strip()

# 3. Check duplicates
print("=== DUPLICATE CHECK ===")
print("AI duplicates:", ai["source"].duplicated().sum())
print("Human duplicates:", human["source"].duplicated().sum())

# 4. Check matching records
ai_sources = set(ai["source"])
human_sources = set(human["source"])

matched = ai_sources & human_sources
ai_only = ai_sources - human_sources
human_only = human_sources - ai_sources

print("\n=== SOURCE MATCHING ===")
print("AI records:", len(ai))
print("Human records:", len(human))
print("Matching records:", len(matched))
print("AI only:", len(ai_only))
print("Human only:", len(human_only))

# 5. Normalize labels
label_map = {
    "1 ปกติ": "ปกติ",
    "2 ไม่ปกติ": "ไม่ปกติ",
    "3 ไม่แน่ใจ": "ไม่แน่ใจ",
    "ปกติ": "ปกติ",
    "ไม่ปกติ": "ไม่ปกติ",
    "ไม่แน่ใจ": "ไม่แน่ใจ"
}

ai["label_normalized"] = ai["label"].map(label_map)
human["label_normalized"] = human["label"].map(label_map)

# 6. Check invalid / missing labels
print("\n=== LABEL CHECK ===")
print("AI invalid labels:")
print(ai[ai["label_normalized"].isna()]["label"].value_counts())

print("\nHuman invalid labels:")
print(human[human["label_normalized"].isna()]["label"].value_counts())

# 7. Merge using source
comparison = pd.merge(
    ai[["source", "label_normalized"]],
    human[["source", "label_normalized"]],
    on="source",
    how="inner",
    suffixes=("_AI", "_Human")
)

# 8. Check records usable for Kappa
valid = comparison.dropna(
    subset=["label_normalized_AI", "label_normalized_Human"]
)

valid = valid[
    (valid["label_normalized_AI"] != "") &
    (valid["label_normalized_Human"] != "")
]

# Exclude AI ERROR / any invalid labels
valid = valid[
    valid["label_normalized_AI"].isin(
        ["ปกติ", "ไม่ปกติ", "ไม่แน่ใจ"]
    )
    &
    valid["label_normalized_Human"].isin(
        ["ปกติ", "ไม่ปกติ", "ไม่แน่ใจ"]
    )
]

print("\n=== FINAL CHECK ===")
print("Matched records:", len(comparison))
print("Valid records for Cohen's Kappa:", len(valid))
print("Agreement:", 
    (valid["label_normalized_AI"] ==
       valid["label_normalized_Human"]).mean() * 100, "%")

# 9. Export comparison
valid.to_csv("AI_vs_Ploy_matched.csv", index=False,
            encoding="utf-8-sig")

print("\nSaved: AI_vs_Ploy_matched.csv")
