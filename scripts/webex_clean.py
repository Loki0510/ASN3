import pandas as pd
import emoji
import os

# File paths
input_path = "data/Webex - Sheet1.csv"
output_path = "data/cleaned_webex_reviews.csv"

# Load data
df = pd.read_csv(input_path)

# Drop irrelevant columns
df = df.drop(columns=["userImage", "replyContent", "repliedAt"])

# Drop rows with missing content or score
df = df.dropna(subset=["content", "score"])

# Normalize text: lowercase and strip whitespace
df["content"] = df["content"].astype(str).str.lower().str.strip()

# Extract emojis
def extract_emojis(text):
    return ''.join(c for c in text if c in emoji.EMOJI_DATA)

df["emojis"] = df["content"].apply(extract_emojis)

# Fill or flag missing versions
df["appVersion"] = df["appVersion"].fillna("unknown")
df["reviewCreatedVersion"] = df["reviewCreatedVersion"].fillna("unknown")

# Drop duplicates
df = df.drop_duplicates(subset=["reviewId", "content"])

# Convert 'at' (timestamp) to datetime
df["at"] = pd.to_datetime(df["at"], errors='coerce')

# Save cleaned dataset
os.makedirs("data", exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Cleaned data saved at: {output_path}")
print(f"Final Shape: {df.shape}")
