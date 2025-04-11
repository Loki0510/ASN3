import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load datasets
zoom_df = pd.read_csv("data/zoom_with_hf_sentiment_sampled.csv")
webex_df = pd.read_csv("data/webex_with_hf_sentiment_sampled.csv")
firefox_df = pd.read_csv("data/firefox_with_hf_sentiment_sampled.csv")

# Combine all three datasets for plotting
zoom_df["app"] = "Zoom"
webex_df["app"] = "Webex"
firefox_df["app"] = "Firefox"

# Concatenate all three datasets
combined_df = pd.concat([zoom_df, webex_df, firefox_df])

# Plot sentiment distribution comparison across apps
plt.figure(figsize=(12, 6))
sns.countplot(x="hf_sentiment_label", hue="app", data=combined_df, palette="Set2")
plt.title("Sentiment Distribution Comparison (Zoom, Webex, Firefox)")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("output/figures/sentiment_comparison_all_apps.png")
plt.close()
