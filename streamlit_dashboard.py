import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Specify the full path to the CSV file
file_path = r'D:\Data Science\AAIT\Cap Stone Project\Data\Merged Data\Fillna\Merged_reviews_cleaned_labeled_topic II.csv'

# Load the reviews data from the specified path
df_labeled = pd.read_csv(file_path)

# Ensure 'Hotel_site' column contains non-null values and is properly formatted as strings
df_labeled['Hotel_site'] = df_labeled['Hotel_site'].astype(str).fillna('Unknown')

# Function to create word-frequency cloud plots
def create_wordcloud(data):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(data))
    fig, ax = plt.subplots(figsize=(10, 6))  
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Streamlit app
st.title("Exploratory Analysis and Dashboard")

# Dropdown for selecting hotel site
hotel_site = st.selectbox('Select Hotel Site', ['All Hotels'] + list(df_labeled['Hotel_site'].unique()))

# Display selected hotel site for debugging
st.write(f"Selected Hotel Site: {hotel_site}")

# Filter data based on selected hotel site
if hotel_site == 'All Hotels':
    filtered_df = df_labeled.copy()
else:
    filtered_df = df_labeled[df_labeled['Hotel_site'] == hotel_site].copy()

# Drop rows with missing Review_comment values
filtered_df = filtered_df.dropna(subset=['Review_comment'])

# Ensure Review_comment column contains strings
filtered_df['Review_comment'] = filtered_df['Review_comment'].astype(str)

# Word Frequency Cloud Plot
st.subheader('Word Cloud of Review Comments')
wordcloud_fig = create_wordcloud(filtered_df['Review_comment'])
st.pyplot(wordcloud_fig)

# Topics of Improvement Areas
st.subheader('Topics of Improvement Areas')
negative_reviews = filtered_df[filtered_df['Sentiment'] == 'Negative']
negative_topics = negative_reviews['Topic_Label'].value_counts()
if not negative_topics.empty:
    fig, ax = plt.subplots(figsize=(10, 6))  
    sns.barplot(x=negative_topics.index, y=negative_topics.values, ax=ax)
    ax.set_xlabel('Topic')
    ax.set_ylabel('Number of Negative Reviews')
    ax.set_title('Topics of Improvement Areas')
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.25)
    st.pyplot(fig)
else:
    st.write("No negative reviews available for the selected hotel site.")

# Sentiment Analysis Plot
st.subheader('Sentiment Analysis')
sentiment_counts = filtered_df['Sentiment'].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))  
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax)
ax.set_xlabel('Sentiment')
ax.set_ylabel('Number of Reviews')
ax.set_title('Sentiment Analysis')
st.pyplot(fig)

# Ratings Plot
st.subheader('Ratings Distribution')
rating_counts = filtered_df['Rating'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 6))  
sns.barplot(x=rating_counts.index, y=rating_counts.values, ax=ax)
ax.set_xlabel('Rating')
ax.set_ylabel('Number of Reviews')
ax.set_title('Ratings Distribution')
st.pyplot(fig)

# Summary Statistics
st.subheader('Summary Statistics')
total_reviews = len(filtered_df)
average_rating = filtered_df['Rating'].mean()
positive_reviews = sentiment_counts.get('Positive', 0)
negative_reviews = sentiment_counts.get('Negative', 0)
neutral_reviews = sentiment_counts.get('Neutral', 0)

# Display summary statistics for debugging
st.write(f"Total Reviews: {total_reviews}")
st.write(f"Average Rating: {average_rating:.2f}")
st.write(f"Number of Positive Reviews: {positive_reviews}")
st.write(f"Number of Negative Reviews: {negative_reviews}")
st.write(f"Number of Neutral Reviews: {neutral_reviews}")
