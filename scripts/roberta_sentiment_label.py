import pandas as pd
import emoji

# Define the emoji sentiment lexicon
emoji_sentiment_map = {
    "😊": "positive", "😍": "positive", "❤": "positive", "👍": "positive", "👌": "positive",
    "😂": "neutral", "😭": "negative", "😢": "negative", "😡": "negative", "😐": "neutral",
    "😩": "negative", "❤️": "positive", "🥺": "neutral", "😎": "positive", "😱": "negative",
    "👎": "negative", "🥴": "negative", "💔": "negative", "💀": "negative", "😜": "positive",
    "🤔": "neutral", "🙏": "neutral", "🤗": "positive"
}

# Function to extract emojis from text
def extract_emojis(text):
    return ''.join(c for c in text if c in emoji.EMOJI_DATA)

# Function to classify emoji sentiment based on the lexicon
def classify_emoji_sentiment(emojis):
    sentiment = "neutral"  # Default sentiment if no match
    for emoji_char in emojis:
        if emoji_char in emoji_sentiment_map:
            sentiment = emoji_sentiment_map[emoji_char]
            break  # Choose the first match (or you could modify this to return the majority sentiment)
    
    return sentiment

# Load the dataset
df = pd.read_csv("data/zoom_with_hf_sentiment_sampled.csv")

# Handle missing values in 'emojis' column and apply emoji sentiment analysis
df['emojis'] = df['emojis'].fillna('')
df['emoji_sentiment'] = df['emojis'].apply(classify_emoji_sentiment)

# Save the updated dataframe with emoji sentiment
df.to_csv("data/zoom_with_emoji_sentiment.csv", index=False)

# Display the updated dataframe with the emoji sentiment column
df[['content', 'emojis', 'emoji_sentiment']].head()
