import streamlit as st
from css import css
from project import data_init
import pandas as pd
import neptune.new as neptune

# hide app menu and footer custiomization
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.subheader("Exploring the data")
#fetch rss data form url 
url = "http://feeds.bbci.co.uk/news/rss.xml"

# Experimental database tracking
run = neptune.init(
    project="jairajpatil8/RssSentiment",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlY2RlZDlmOS1jNmIzLTQ1ZjQtYTBjMi0xNjUzZmY1ZjZmY2QifQ==",
)  

data_df = data_init.get_rss_feed(url)
run["new_data_rows"] = data_df.shape[0]
run["new_data_columns"] = data_df.shape[1]

# Load data from database and concatenate old and new data
old_data_df = pd.read_csv("bbc_news.csv")
run["old_data_rows"] = old_data_df.shape[0]
run["old_data_columns"] = old_data_df.shape[1]

# Merge the newly parsed data with existing one. Remove duplicates.
new_data_df = pd.concat([old_data_df, data_df], axis=0)
new_data_df = new_data_df.drop_duplicates()
run["merged_data_rows"] = new_data_df.shape[0]
run["merged_data_columns"] = new_data_df.shape[1]


# Save merged data
new_data_df.to_csv("bbc_news.csv", index=False)
run.stop()
#BBC
my_expander = st.expander(label='Show updated records for BBC News')
with my_expander:
	st.subheader("Rss Feeds BBC News Data")
	st.write(f"New data collected: {data_df.shape[0]}")
	st.write(data_df.head())

	st.write(f"old data: {old_data_df.shape[0]}")
	st.write(old_data_df.head())

	st.write(f"Data after concatenation: {new_data_df.shape[0]}")
	st.write(f"Data after droping duplicates: {new_data_df.shape[0]}")

#TOI
url = "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"

data_df = data_init.get_rss_feed(url)
# run["new_data_rows"] = data_df.shape[0]
# run["new_data_columns"] = data_df.shape[1]
# print(f"New data collected: {data_df.shape[0]}")
data_df.head()

old_data_df = pd.read_csv("toi_news.csv")
# run["old_data_rows"] = old_data_df.shape[0]
# run["old_data_columns"] = old_data_df.shape[1]
# print(f"Old data: {old_data_df.shape[0]}")
old_data_df.head()

new_data_df = pd.concat([old_data_df, data_df], axis=0)
# print(f"Data after concatenation: {new_data_df.shape[0]}")
new_data_df = new_data_df.drop_duplicates()
# run["merged_data_rows"] = new_data_df.shape[0]
# run["merged_data_columns"] = new_data_df.shape[1]
# print(f"Data after droping duplicates: {new_data_df.shape[0]}")
new_data_df.head()

new_data_df.to_csv("toi_news.csv", index=False)


my_expander = st.expander(label='Show updated records for TOI News')
with my_expander:
	st.subheader("Rss Feeds TOI News Data")
	st.write(f"New data collected: {data_df.shape[0]}")
	st.write(data_df.head())

	st.write(f"old data: {old_data_df.shape[0]}")
	st.write(old_data_df.head())

	st.write(f"Data after concatenation: {new_data_df.shape[0]}")
	st.write(f"Data after droping duplicates: {new_data_df.shape[0]}")

#NYTimes

url = "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"

data_df = data_init.get_rss_feed(url)
# run["new_data_rows"] = data_df.shape[0]
# run["new_data_columns"] = data_df.shape[1]
# print(f"New data collected: {data_df.shape[0]}")
data_df.head()

old_data_df = pd.read_csv("ny_news.csv")
# run["old_data_rows"] = old_data_df.shape[0]
# run["old_data_columns"] = old_data_df.shape[1]
# print(f"Old data: {old_data_df.shape[0]}")
old_data_df.head()

new_data_df = pd.concat([old_data_df, data_df], axis=0)
# print(f"Data after concatenation: {new_data_df.shape[0]}")
new_data_df = new_data_df.drop_duplicates()
# run["merged_data_rows"] = new_data_df.shape[0]
# run["merged_data_columns"] = new_data_df.shape[1]
# print(f"Data after droping duplicates: {new_data_df.shape[0]}")
new_data_df.head()

new_data_df.to_csv("ny_news.csv", index=False)

my_expander = st.expander(label='Show updated records for NewYork TImes')
with my_expander:

	st.subheader("Rss Feeds NYTimes News Data")
	st.write(f"New data collected: {data_df.shape[0]}")
	st.write(data_df.head())

	st.write(f"old data: {old_data_df.shape[0]}")
	st.write(old_data_df.head())

	st.write(f"Data after concatenation: {new_data_df.shape[0]}")
	st.write(f"Data after droping duplicates: {new_data_df.shape[0]}")

