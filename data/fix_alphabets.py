import pandas as pd
from ftfy import fix_text
import unicodedata

def to_ascii(text):
    if not isinstance(text, str):
        return text

    # Fix broken encoding (Ã© → é)
    text = fix_text(text)

    # Remove accents (é → e)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    return text


# Read CSV safely
df = pd.read_csv("kreol_toxicity.csv", encoding="latin1")

# Apply to ALL text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].apply(to_ascii)

# Save cleaned CSV
df.to_csv("kreol_toxicity_ascii.csv", index=False, encoding="utf-8")

print("✅ Done! All text converted to plain ASCII")
