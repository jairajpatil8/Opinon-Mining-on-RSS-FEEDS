import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import warnings
from project import data_init
warnings.simplefilter("ignore")

st.set_option('deprecation.showPyplotGlobalUse', False)
# hide app menu and footer custiomization
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


#  #Topic modeling helper functions

#         # Define helper functions
# def get_keys(topic_matrix):
#     '''
#     returns an integer list of predicted topic 
#     categories for a given topic matrix
#     '''
#     keys = topic_matrix.argmax(axis=1).tolist()
#     return keys
# def keys_to_counts(keys):
#     '''
#     returns a tuple of topic categories and their 
#     accompanying magnitudes for a given list of keys
#     '''
#     count_pairs = Counter(keys).items()
#     categories = [pair[0] for pair in count_pairs]
#     counts = [pair[1] for pair in count_pairs]
#     return (categories, counts)

# # Define helper functions
# def get_top_n_words(n, keys, document_term_matrix, count_vectorizer):
#     '''
#     returns a list of n_topic strings, where each string contains the n most common 
#     words in a predicted category, in order
#     '''
#     top_word_indices = []
#     for topic in range(n_topics):
#         temp_vector_sum = 0
#         for i in range(len(keys)):
#             if keys[i] == topic:
#                 temp_vector_sum += document_term_matrix[i]
#         temp_vector_sum = temp_vector_sum.toarray()
#         top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:],0)
#         top_word_indices.append(top_n_word_indices)   
#     top_words = []
#     for topic in top_word_indices:
#         topic_words = []
#         for index in topic:
#             temp_word_vector = np.zeros((1,document_term_matrix.shape[1]))
#             temp_word_vector[:,index] = 1
#             the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]
#             topic_words.append(the_word.encode('ascii').decode('utf-8'))
#         top_words.append(" ".join(topic_words))         
#     return top_words


score_data=pd.DataFrame()

# BBC
st.subheader("BBC News Feeds")
my_expander = st.expander(label='Analyze text data')
with my_expander:

	st.markdown("We quickly look to the text data by showing wordclouds of `title` and `description` fields.")
	data_df = pd.read_csv("bbc_news.csv")
	# Title wordcloud
	st.subheader(" Prevalent words in Title")
	data_init.show_wordcloud(data_df['title'], title = '', mask=None)
	# Description WordCloud
	st.subheader(" Prevalent words in Description")
	data_init.show_wordcloud(data_df['description'], title = '', mask=None)


my_expander = st.expander(label='sentiment Bar Plot')
with my_expander:
    data_df['description_score'] = data_df['description'].apply(lambda x: data_init.find_score(x))
    data_df['title_score'] = data_df['title'].apply(lambda x: data_init.find_score(x))
    score_data['BBC_description_score']=data_df['title_score']
    score_data['BBC_title_score']=data_df['description_score']
	# description sentiment
    data_df['description_sentiment'] = data_df['description'].apply(lambda x: data_init.find_sentiment(x))
    data_init.plot_sentiment(data_df, 'description_sentiment', 'Description')
    
	# Title sentiment
    data_df['title_sentiment'] = data_df['title'].apply(lambda x: data_init.find_sentiment(x))
    data_init.plot_sentiment(data_df, 'title_sentiment', 'Title')


my_expander = st.expander(label='WordCloud based on word Sentiment')
with my_expander:
	st.subheader("Wordclouds with description words, grouped by sentiment")
	data_init.show_wordcloud(data_df.loc[data_df['description_sentiment']=="Positive", 'description'], title = 'Prevalent words in Description with Positive sentiment', mask=None)
	data_init.show_wordcloud(data_df.loc[data_df['description_sentiment']=="Negative", 'description'], title = 'Prevalent words in Description with Negative sentiment', mask=None)
	data_init.show_wordcloud(data_df.loc[data_df['description_sentiment']=="Neutral", 'description'], title = 'Prevalent words in Description with Neutral sentiment', mask=None)

	st.subheader("Wordclouds with title words, grouped by sentiment")
	data_init.show_wordcloud(data_df.loc[data_df['title_sentiment']=="Positive", 'title'], title = 'Prevalent words in Title with Positive sentiment', mask=None)
	data_init.show_wordcloud(data_df.loc[data_df['title_sentiment']=="Negative", 'title'], title = 'Prevalent words in Title with Negative sentiment', mask=None)
	data_init.show_wordcloud(data_df.loc[data_df['title_sentiment']=="Neutral", 'title'], title = 'Prevalent words in Title with Neutral sentiment', mask=None)

# my_expander = st.expander(label='Topic Modeling')
# with my_expander:
# 	# text vectorization
# 	count_vectorizer = CountVectorizer(stop_words='english', max_features=40000)
# 	# drop rows without text
# 	sample_text_df = data_df[["description"]].dropna()
# 	text_sample = sample_text_df["description"].sample(n=sample_text_df.shape[0], random_state=0).values

# 	# st.write('Text sample vectorization: {}'.format(text_sample[9]))
# 	document_term_matrix = count_vectorizer.fit_transform(text_sample)
# 	# st.write('Text sample vectorization: \n{}'.format(document_term_matrix[9]))

# 	n_topics = 10
# 	lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online', random_state=0, verbose=0)
# 	lda_topic_matrix = lda_model.fit_transform(document_term_matrix)

# 	lsa_keys = get_keys(lda_topic_matrix)
# 	lsa_categories, lsa_counts = keys_to_counts(lsa_keys)

# 	top_n_words_lsa =get_top_n_words(n_topics, lsa_keys, document_term_matrix, count_vectorizer)

# 	for i in range(len(top_n_words_lsa)):
# 		st.write("Topic {}: ".format(i+1),top_n_words_lsa[i])


# TOI
st.subheader("Times of India News")
my_expander = st.expander(label='Analyze text data')
with my_expander:

    st.markdown("We quickly look to the text data by showing wordclouds of `title` and `description` fields.")
    data_df = pd.read_csv("toi_news.csv")
    # Title wordcloud
    st.subheader(" Prevalent words in Title")
    data_init.show_wordcloud(data_df['title'], title = '', mask=None)
    # Description WordCloud
    st.subheader(" Prevalent words in Description")
    data_init.show_wordcloud(data_df['description'], title = '', mask=None)


my_expander = st.expander(label='sentiment Bar Plot')
with my_expander:
    data_df['description_score'] = data_df['description'].apply(lambda x: data_init.find_score(x))
    data_df['title_score'] = data_df['title'].apply(lambda x: data_init.find_score(x))
    score_data['toi_description_score']=data_df['title_score']
    score_data['toi_title_score']=data_df['description_score']
    # description sentiment
    data_df['description_sentiment'] = data_df['description'].apply(lambda x: data_init.find_sentiment(x))
    data_init.plot_sentiment(data_df, 'description_sentiment', 'Description')
    
    # Title sentiment
    data_df['title_sentiment'] = data_df['title'].apply(lambda x: data_init.find_sentiment(x))
    data_init.plot_sentiment(data_df, 'title_sentiment', 'Title')


my_expander = st.expander(label='WordCloud based on word Sentiment')
with my_expander:
    st.subheader("Wordclouds with description words, grouped by sentiment")
    data_init.show_wordcloud(data_df.loc[data_df['description_sentiment']=="Positive", 'description'], title = 'Prevalent words in Description with Positive sentiment', mask=None)
    data_init.show_wordcloud(data_df.loc[data_df['description_sentiment']=="Negative", 'description'], title = 'Prevalent words in Description with Negative sentiment', mask=None)
    data_init.show_wordcloud(data_df.loc[data_df['description_sentiment']=="Neutral", 'description'], title = 'Prevalent words in Description with Neutral sentiment', mask=None)

    st.subheader("Wordclouds with title words, grouped by sentiment")
    data_init.show_wordcloud(data_df.loc[data_df['title_sentiment']=="Positive", 'title'], title = 'Prevalent words in Title with Positive sentiment', mask=None)
    data_init.show_wordcloud(data_df.loc[data_df['title_sentiment']=="Negative", 'title'], title = 'Prevalent words in Title with Negative sentiment', mask=None)
    data_init.show_wordcloud(data_df.loc[data_df['title_sentiment']=="Neutral", 'title'], title = 'Prevalent words in Title with Neutral sentiment', mask=None)

# my_expander = st.expander(label='Topic Modeling')
# with my_expander:
#     # text vectorization
#     count_vectorizer = CountVectorizer(stop_words='english', max_features=40000)
#     # drop rows without text
#     sample_text_df = data_df[["description"]].dropna()
#     text_sample = sample_text_df["description"].sample(n=sample_text_df.shape[0], random_state=0).values

#     # st.write('Text sample vectorization: {}'.format(text_sample[9]))
#     document_term_matrix = count_vectorizer.fit_transform(text_sample)
#     # st.write('Text sample vectorization: \n{}'.format(document_term_matrix[9]))

#     n_topics = 10
#     lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online', random_state=0, verbose=0)
#     lda_topic_matrix = lda_model.fit_transform(document_term_matrix)

#     lsa_keys = get_keys(lda_topic_matrix)
#     lsa_categories, lsa_counts = keys_to_counts(lsa_keys)

#     top_n_words_lsa = get_top_n_words(n_topics, lsa_keys, document_term_matrix, count_vectorizer)

#     for i in range(len(top_n_words_lsa)):
#         st.write("Topic {}: ".format(i+1),top_n_words_lsa[i])



# NY Times
st.subheader("New York Times")
my_expander = st.expander(label='Analyze text data')
with my_expander:

    st.markdown("We quickly look to the text data by showing wordclouds of `title` and `description` fields.")
    data_df = pd.read_csv("ny_news.csv")
    # Title wordcloud
    st.subheader(" Prevalent words in Title")
    data_init.show_wordcloud(data_df['title'], title = '', mask=None)
    # Description WordCloud
    st.subheader(" Prevalent words in Description")
    data_init.show_wordcloud(data_df['description'], title = '', mask=None)


my_expander = st.expander(label='sentiment Bar Plot')
with my_expander:
    data_df['description_score'] = data_df['description'].apply(lambda x: data_init.find_score(x))
    data_df['title_score'] = data_df['title'].apply(lambda x: data_init.find_score(x))
    score_data['NYT_description_score']=data_df['title_score']
    score_data['NYT_title_score']=data_df['description_score']
    # description sentiment
    data_df['description_sentiment'] = data_df['description'].apply(lambda x: data_init.find_sentiment(x))
    data_init.plot_sentiment(data_df, 'description_sentiment', 'Description')

    # Title sentiment
    data_df['title_sentiment'] = data_df['title'].apply(lambda x: data_init.find_sentiment(x))
    data_init.plot_sentiment(data_df, 'title_sentiment', 'Title')


my_expander = st.expander(label='WordCloud based on word Sentiment')
with my_expander:
    st.subheader("Wordclouds with description words, grouped by sentiment")
    data_init.show_wordcloud(data_df.loc[data_df['description_sentiment']=="Positive", 'description'], title = 'Prevalent words in Description with Positive sentiment', mask=None)
    data_init.show_wordcloud(data_df.loc[data_df['description_sentiment']=="Negative", 'description'], title = 'Prevalent words in Description with Negative sentiment', mask=None)
    data_init.show_wordcloud(data_df.loc[data_df['description_sentiment']=="Neutral", 'description'], title = 'Prevalent words in Description with Neutral sentiment', mask=None)

    st.subheader("Wordclouds with title words, grouped by sentiment")
    data_init.show_wordcloud(data_df.loc[data_df['title_sentiment']=="Positive", 'title'], title = 'Prevalent words in Title with Positive sentiment', mask=None)
    data_init.show_wordcloud(data_df.loc[data_df['title_sentiment']=="Negative", 'title'], title = 'Prevalent words in Title with Negative sentiment', mask=None)
    data_init.show_wordcloud(data_df.loc[data_df['title_sentiment']=="Neutral", 'title'], title = 'Prevalent words in Title with Neutral sentiment', mask=None)

# my_expander = st.expander(label='Topic Modeling')
# with my_expander:
#     # text vectorization
#     count_vectorizer = CountVectorizer(stop_words='english', max_features=40000)
#     # drop rows without text
#     sample_text_df = data_df[["description"]].dropna()
#     text_sample = sample_text_df["description"].sample(n=sample_text_df.shape[0], random_state=0).values

#     # st.write('Text sample vectorization: {}'.format(text_sample[9]))
#     document_term_matrix = count_vectorizer.fit_transform(text_sample)
#     # st.write('Text sample vectorization: \n{}'.format(document_term_matrix[9]))

#     n_topics = 10
#     lda_model = LatentDirichletAllocation(n_components=n_topics, learning_method='online', random_state=0, verbose=0)
#     lda_topic_matrix = lda_model.fit_transform(document_term_matrix)

#     lsa_keys = get_keys(lda_topic_matrix)
#     lsa_categories, lsa_counts =keys_to_counts(lsa_keys)

#     top_n_words_lsa = get_top_n_words(n_topics, lsa_keys, document_term_matrix, count_vectorizer)

#     for i in range(len(top_n_words_lsa)):
#         st.write("Topic {}: ".format(i+1),top_n_words_lsa[i])

# score_data.to_csv("Hscores_news.csv", index=False)
# Hdata_df = pd.read_csv("Hscores_news.csv")
# st.area_chart(Hdata_df)