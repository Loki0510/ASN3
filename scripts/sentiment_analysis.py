import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Prepare output folder
os.makedirs("output/figures", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("data/cleaned_zoom_reviews.csv")
df["at"] = pd.to_datetime(df["at"])

# --- Sentiment Analysis Using VADER ---
analyzer = SentimentIntensityAnalyzer()

df["sentiment_score"] = df["content"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
df["sentiment_label"] = df["sentiment_score"].apply(
    lambda score: "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
)

# --- Emoji Polarity Mapping ---
positive_emojis = set("😊😍😁😃😄👍🔥🥰✨❤️🤩")
negative_emojis = set("😡😠😢😭👎😞💔🤬😣")

def emoji_sentiment(emojis):
    if pd.isna(emojis) or not emojis.strip():
        return "neutral"
    pos = sum(1 for e in emojis if e in positive_emojis)
    neg = sum(1 for e in emojis if e in negative_emojis)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    else:
        return "neutral"

df["emoji_sentiment"] = df["emojis"].apply(emoji_sentiment)

# Save updated dataset
df.to_csv("data/zoom_with_sentiment.csv", index=False)

# --- Visualizations ---

# Sentiment over time
sentiment_trend = df.groupby(df["at"].dt.to_period("M"))["sentiment_score"].mean()

plt.figure(figsize=(12, 6))
sentiment_trend.plot()
plt.title("Average Sentiment Over Time")
plt.ylabel("Sentiment Score")
plt.xlabel("Month")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/figures/sentiment_over_time.png")
plt.show()

# Emoji Sentiment Pie
plt.figure(figsize=(6, 6))
df["emoji_sentiment"].value_counts().plot.pie(autopct="%1.1f%%", colors=["lightgreen", "lightcoral", "lightgrey"])
plt.title("Emoji Sentiment Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig("output/figures/emoji_sentiment_pie.png")
plt.show()

# Sentiment vs Emoji Polarity Heatmap (optional)
heatmap_data = pd.crosstab(df["sentiment_label"], df["emoji_sentiment"])
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm")
plt.title("Text Sentiment vs Emoji Polarity")
plt.tight_layout()
plt.savefig("output/figures/sentiment_vs_emoji_heatmap.png")
plt.show()
