import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page configuration
st.set_page_config(page_title="Zoom, Webex & Firefox Review Insights", layout="wide")

# Load data for each dataset
@st.cache_data
def load_data(file_name):
    return pd.read_csv(file_name)

zoom_data = load_data("data/zoom_with_hf_sentiment_sampled.csv")
webex_data = load_data("data/webex_with_hf_sentiment_sampled.csv")
firefox_data = load_data("data/firefox_with_hf_sentiment_sampled.csv")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Sentiment Analysis",
    "General Insights",
    "Trend Analysis",
    "Interactive KPI Explorer"
])

# Home
if page == "Home":
    st.title("📊 Zoom, Webex & Firefox Review Insights Dashboard")
    st.markdown("""
    Welcome to the **App Review Insights Dashboard**! Here, we explore feedback and sentiment analysis for **Zoom**, **Webex**, and **Firefox** apps.
    
    The main goal of this design is to **map user feedback to app versions**, focusing on **emoji sentiment** and the **evolution of app versions** through user reviews. By analyzing the sentiment of **3,750 sampled reviews per app** (Zoom, Webex, and Firefox), we gain insights into how users perceive the changes made to these apps.

    ### Why did we choose 3,750 reviews per app?
    We selected **3,750 reviews** to represent a **diverse range** of user experiences without overwhelming the analysis. This sample size gives us a **balanced view** of user sentiment, focusing on relevant periods with **stratified sampling** to cover important trends.

    ### Why VADER and RoBERTa?
    We initially applied **VADER** sentiment analysis, but we noticed it had **limitations in capturing nuanced emotions** in reviews, especially in detecting **positive and negative sentiment shifts**. To improve accuracy, we shifted to **RoBERTa**, a more powerful model that better understands the emotional complexity of user feedback.

    ### What you will see in the dashboard:
    - **General Insights**: Review trends over time, rating distribution, top emojis, and most common words.
    - **Sentiment Analysis**: Compare VADER and RoBERTa sentiment analysis for each app, along with their improvements.
    - **Trend Analysis**: Compare sentiment and emoji sentiment trends across **Zoom**, **Webex**, and **Firefox**.

    Let's dive into the insights!
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Zoom Reviews", 3750)  # As per your choice of 3750 reviews
    col2.metric("Webex Reviews", 3750)
    col3.metric("Firefox Reviews", 3750)
    
    st.image("output/figures/sentiment_comparison_all_apps.png", caption="Sentiment Analysis Across all Apps", use_container_width=True)

# Sentiment Analysis
elif page == "Sentiment Analysis":
    sentiment_page = st.sidebar.radio("Sentiment Analysis for:", ["Zoom VADER", "Zoom RoBERTa", "Webex Sentiment", "Firefox Sentiment"])
    
    if sentiment_page == "Zoom VADER":
        st.title("Zoom VADER Sentiment Analysis")
        st.image("output/figures/sentiment_over_time.png", caption="VADER Sentiment Over Time", use_container_width=True)
        st.image("output/figures/emoji_sentiment_pie.png", caption="Emoji Sentiment Distribution (VADER-Based)", use_container_width=True)
        st.image("output/figures/sentiment_vs_emoji_heatmap.png", caption="Text Sentiment vs Emoji Sentiment (VADER-Based)", use_container_width=True)
        
        st.markdown("""The **VADER model** exhibits a **neutral bias** when analyzing sentiment, as seen in the **Sentiment Over Time** and **Emoji Sentiment Pie** charts. 
        This is why we moved to **RoBERTa**, which offers better understanding of emotional signals.
        """)
        
    elif sentiment_page == "Zoom RoBERTa":
        st.title("Zoom RoBERTa Sentiment Analysis")
        st.image("output/figures/roberta_sentiment_bar.png", caption="Webex Sentiment Distribution", use_container_width=True)
        st.image("output/figures/roberta_sentiment_pie.png", caption="Webex Sentiment Proportion", use_container_width=True)
        st.image("output/figures/roberta_sentiment_over_time.png", caption="Webex Monthly Sentiment Trends", use_container_width=True)
        
        st.markdown("""**RoBERTa Sentiment Analysis** provides a more **accurate sentiment breakdown** compared to VADER.
        The **RoBERTa Sentiment Over Time** chart shows a clear evolution of sentiment, providing a deeper understanding of user feedback over different app versions.
        """)
    
    elif sentiment_page == "Webex Sentiment":
        st.title("Webex Sentiment Analysis")
        st.image("output/figures/webex_roberta_sentiment_bar.png", caption="Webex Sentiment Distribution", use_container_width=True)
        st.image("output/figures/webex_roberta_sentiment_pie.png", caption="Webex Sentiment Proportion", use_container_width=True)
        st.image("output/figures/webex_roberta_sentiment_over_time.png", caption="Webex Monthly Sentiment Trends", use_container_width=True)
        
        st.markdown("""The **Webex sentiment model** shows more **nuanced polarity** than VADER, with better distinction between positive, neutral, and negative sentiments.
        The **Sentiment Trends** chart indicates fluctuation in user sentiment, potentially tied to app updates and feature releases.
        """)
        
    elif sentiment_page == "Firefox Sentiment":
        st.title("Firefox Sentiment Analysis")
        st.image("output/figures/firefox_roberta_sentiment_bar.png", caption="Firefox Sentiment Distribution", use_container_width=True)
        st.image("output/figures/firefox_roberta_sentiment_pie.png", caption="Firefox Sentiment Proportion", use_container_width=True)
        st.image("output/figures/firefox_roberta_sentiment_over_time.png", caption="Firefox Monthly Sentiment Trends", use_container_width=True)
        
        st.markdown("""The **Sentiment Distribution** and **Monthly Trends** for Firefox indicate **clear shifts** in sentiment over time.
        The **Pie Chart** indicates a well-balanced sentiment between positive, neutral, and negative reviews.
        """)

# General Insights
elif page == "General Insights":
    general_page = st.sidebar.radio("General Insights for:",["Zoom ", "Webex ", "Firefox "])

    if general_page == "Zoom":
        st.title("Zoom General Insights")
        st.image("output/figures/review_count_over_time.png", caption="Zoom Review Count Over Time", use_container_width=True)
        st.image("output/figures/rating_distribution.png", caption="Zoom Rating Distribution", use_container_width=True)
        st.image("output/figures/wordcloud_reviews.png", caption="Zoom Most Common Words in Reviews", use_container_width=True)
        st.image("output/figures/top_emojis.png", caption="Zoom Top 20 Emojis", use_container_width=True)

    elif general_page == "Webex" :
        st.title("Webex General Insights")
        st.image("output/figures/webex_review_count_over_time.png", caption="Webex Review Count Over Time", use_container_width=True)
        st.image("output/figures/webex_rating_distribution.png", caption="Webex Rating Distribution", use_container_width=True)
        st.image("output/figures/webex_word_cloud.png", caption="Webex Most Common Words in Reviews", use_container_width=True)
        st.image("output/figures/webex_top_emojis.png", caption="Webex Top 20 Emojis", use_container_width=True)

    elif general_page == "Firefox" :
        st.title("Firefox General Insights")
        st.image("output/figures/firefox_review_count_over_time.png", caption="Firefox Review Count Over Time", use_container_width=True)
        st.image("output/figures/firefox_rating_distribution.png", caption="Firefox Rating Distribution", use_container_width=True)
        st.image("output/figures/firefox_word_cloud.png", caption="Firefox Most Common Words in Reviews", use_container_width=True)
        st.image("output/figures/firefox_top_emojis.png", caption="Firefox Top 20 Emojis", use_container_width=True)

# Trend Analysis
elif page == "Trend Analysis":
    st.title("📈 Trend Analysis Across Apps")
    st.markdown("""
    In this section, we compare sentiment trends, emoji sentiment, and ratings for all three apps.
    We'll show how user engagement and emotional responses change over time for **Zoom**, **Webex**, and **Firefox**.
    """)
    
    st.image("output/figures/zoom_sentiment_over_time_by_version.png", caption="Zoom Sentiment Trend Comparison", use_container_width=True)
    st.image("output/figures/webex_sentiment_over_time_by_version.png", caption="Webex Sentiment Trend Comparison", use_container_width=True)
    st.image("output/figures/firefox_sentiment_over_time_by_version.png", caption="Firefox Sentiment Trend Comparison", use_container_width=True)
    
    st.markdown("""
    These trend comparisons allow us to observe how sentiment evolves over time for each of the apps. The fluctuations in user sentiment can often be tied to new app releases, features, or updates.
    """)


# Interactive KPI Explorer
elif page == "Interactive KPI Explorer":
    st.title("📊 Interactive Plot Generator")
    st.markdown("""
    Here, you can interactively explore the data by uploading the dataset and selecting different columns for dynamic plot generation. This will help in exploring specific relationships between features and understanding deeper insights.
    """)
    
    # File upload option
    uploaded_file = st.file_uploader("Upload Sentiment Analysis Dataset (Zoom, Webex, or Firefox)", type=["csv"])

    if uploaded_file is not None:
        # Load the uploaded dataset
        df = pd.read_csv(uploaded_file)
        
        # Display some basic information about the dataset
        st.write(f"Data Loaded: {uploaded_file.name}")
        st.write(f"Shape of Data: {df.shape}")
        st.write("Preview of the Data:")
        st.write(df.head())

        # Suggest X-axis and Y-axis columns based on the dataset
        if 'hf_sentiment_label' in df.columns:
            x_axis_options = ['at', 'appVersion', 'score', 'hf_sentiment_label']
            y_axis_options = ['score', 'hf_sentiment_label', 'reviewCount']
        else:
            x_axis_options = df.columns.tolist()
            y_axis_options = df.columns.tolist()
        
        # Provide suggestions for good plots
        st.markdown("""
        **Suggested Plots:**
        - **Review Count Over Time**: Use `at` for X-axis and `reviewCount` for Y-axis.
        - **Sentiment vs. Rating**: Use `hf_sentiment_label` for X-axis and `score` for Y-axis.
        - **Emoji Sentiment Distribution**: Use `hf_sentiment_label` for X-axis and `emoji_sentiment` for Y-axis.
        """)

        # Select X and Y axes for the plot
        x_axis = st.selectbox("Select X-axis", x_axis_options)
        y_axis = st.selectbox("Select Y-axis", y_axis_options)

        # Plot type dropdown
        plot_type = st.selectbox("Select Plot Type", ["Line Plot", "Bar Plot", "Scatter Plot", "Pie Chart", "Heatmap"])

        # Generate Plot
        if st.button("Generate Plot"):
            fig, ax = plt.subplots()

            if plot_type == "Line Plot":
                ax.plot(df[x_axis], df[y_axis], marker='o')
            elif plot_type == "Bar Plot":
                ax.bar(df[x_axis], df[y_axis])
            elif plot_type == "Scatter Plot":
                ax.scatter(df[x_axis], df[y_axis])
            elif plot_type == "Pie Chart":
                df[y_axis].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                ax.set_ylabel('')
            elif plot_type == "Heatmap":
                pivot = pd.pivot_table(df, values=y_axis, index=x_axis, aggfunc='mean')
                sns.heatmap(pivot, ax=ax, cmap='coolwarm')

            ax.set_title(f"{plot_type}: {x_axis} vs {y_axis}")
            plt.xticks(rotation=45)
            st.pyplot(fig)
