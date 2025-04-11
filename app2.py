import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load data
df = pd.read_csv('data/zoom_with_emoji_sentiment.csv')

# Sidebar for version selection
st.sidebar.header('Filter by Version')
version_options = df['reviewCreatedVersion'].unique().tolist()
selected_version = st.sidebar.selectbox('Select App Version:', version_options)

# Filter dataset based on selected version
filtered_df = df[df['reviewCreatedVersion'] == selected_version]

# --- Sentiment Distribution Comparison (Review vs Emoji) ---
st.header('Sentiment Distribution Comparison (Review vs Emoji)')
sentiment_comparison = pd.DataFrame({
    'Sentiment': ['Positive', 'Neutral', 'Negative'],
    'Review Content': [filtered_df[filtered_df['hf_sentiment_label'] == 'positive'].shape[0],
                       filtered_df[filtered_df['hf_sentiment_label'] == 'neutral'].shape[0],
                       filtered_df[filtered_df['hf_sentiment_label'] == 'negative'].shape[0]],
    'Emoji Sentiment': [filtered_df[filtered_df['emoji_sentiment'] == 'positive'].shape[0],
                        filtered_df[filtered_df['emoji_sentiment'] == 'neutral'].shape[0],
                        filtered_df[filtered_df['emoji_sentiment'] == 'negative'].shape[0]]
})

sentiment_comparison_melt = sentiment_comparison.melt(id_vars='Sentiment', value_vars=['Review Content', 'Emoji Sentiment'], 
                                                     var_name='Sentiment Type', value_name='Count')

fig = px.bar(sentiment_comparison_melt, x='Sentiment', y='Count', color='Sentiment Type', 
             barmode='group', labels={'Sentiment': 'Sentiment Type', 'Count': 'Number of Reviews'})
st.plotly_chart(fig)

# --- Sentiment Agreement Scatter Plot ---
st.header('Sentiment Agreement')
agreement_df = filtered_df[['hf_sentiment_label', 'emoji_sentiment']].dropna()
agreement_df['Agreement'] = agreement_df['hf_sentiment_label'] == agreement_df['emoji_sentiment']

# Plotting sentiment agreement
fig = px.scatter(agreement_df, x='hf_sentiment_label', y='emoji_sentiment', color='Agreement', 
                 title='Sentiment Agreement: Text vs. Emoji')
st.plotly_chart(fig)

# --- Emoji Distribution Plot ---
st.header('Top Emojis and Their Sentiment')
emoji_count = filtered_df['emojis'].explode().value_counts().reset_index()
emoji_count.columns = ['Emoji', 'Count']

# Map sentiment to emojis
emoji_count['Sentiment'] = emoji_count['Emoji'].map(lambda emoji: emoji_sentiment_map.get(emoji, 'neutral'))

fig = px.bar(emoji_count, x='Emoji', y='Count', color='Sentiment', 
             title='Top Emojis and Their Sentiment', labels={'Emoji': 'Emoji', 'Count': 'Frequency'})
st.plotly_chart(fig)

# --- Sentiment Over Time (Frustration Indicator) ---
st.header('Sentiment Over Time')
filtered_df['Month'] = pd.to_datetime(filtered_df['at']).dt.to_period('M')
monthly_sentiment = filtered_df.groupby('Month').agg(
    text_sentiment_count=('hf_sentiment_label', lambda x: x.value_counts().get('positive', 0)),
    emoji_sentiment_count=('emoji_sentiment', lambda x: x.value_counts().get('positive', 0))
).reset_index()

fig = px.line(monthly_sentiment, x='Month', y=['text_sentiment_count', 'emoji_sentiment_count'], 
              labels={'value': 'Count', 'Month': 'Time'},
              title='Sentiment Over Time: Review vs Emoji')
st.plotly_chart(fig)

# --- Drill-Down View ---
st.header('Sample Reviews')
sample_reviews = filtered_df[['content', 'emojis', 'hf_sentiment_label', 'emoji_sentiment']].sample(5)
st.write(sample_reviews)
