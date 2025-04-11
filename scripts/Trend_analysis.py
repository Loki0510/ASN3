import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data for each app
zoom_df = pd.read_csv("data/zoom_with_hf_sentiment_sampled.csv")
webex_df = pd.read_csv("data/webex_with_hf_sentiment_sampled.csv")
firefox_df = pd.read_csv("data/firefox_with_hf_sentiment_sampled.csv")

# Convert the 'at' column to datetime for time-based plotting
zoom_df["at"] = pd.to_datetime(zoom_df["at"], errors="coerce")
webex_df["at"] = pd.to_datetime(webex_df["at"], errors="coerce")
firefox_df["at"] = pd.to_datetime(firefox_df["at"], errors="coerce")

# Set the output directory for saved plots
output_dir = "output/figures"
os.makedirs(output_dir, exist_ok=True)

# Function to generate Sentiment Over Time by App Version plot
def generate_sentiment_over_time_by_version(df, app_name):
    # Create a new column for period (monthly granularity)
    df["month"] = df["at"].dt.to_period("M")

    # Group by app version and sentiment, and count occurrences
    sentiment_counts = df.groupby([df["month"], "appVersion", "hf_sentiment_label"]).size().unstack().fillna(0)

    # Plot sentiment distribution over time by app version
    plt.figure(figsize=(12, 6))
    sentiment_counts.plot(kind="line", marker="o", figsize=(12, 6))
    plt.title(f"Sentiment Over Time by Version ({app_name})")
    plt.xlabel("Month")
    plt.ylabel("Review Count")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    output_file = f"{output_dir}/{app_name.lower()}_sentiment_over_time_by_version.png"
    plt.savefig(output_file)
    plt.close()
    print(f"✅ {app_name} Sentiment Over Time by Version plot saved to: {output_file}")

# Generate plots for each app
generate_sentiment_over_time_by_version(zoom_df, "Zoom")
generate_sentiment_over_time_by_version(webex_df, "Webex")
generate_sentiment_over_time_by_version(firefox_df, "Firefox")
