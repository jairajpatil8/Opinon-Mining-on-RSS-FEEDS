import streamlit as st
from css import css
from project import data_init
import pandas as pd




def main():
	st.title("Opinoin mining on Rss News feeds😍")
	st.markdown("Comparative analysis!!!")
	#st.sidebar.title("sidebar")
	# score_df= pd.DataFrame(columns = ['toi_dec_sent', 'toi_title_sent', 'BBC_title_sent', 'BBC_dec_sent', 'ny_title_sent','ny_dec_sent'])
	my_expander = st.expander(label='Recent TOI feeds with sentiment and polarity scores')
	score_df=pd.DataFrame()
	with my_expander:
		#TOI
		url = "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"

		count=5
		index=1
		top_df =pd.DataFrame()
		top_df = data_init.get_rss_feed(url)
		top_df = top_df.drop(['pubDate', 'guid','link'], axis=1)
		top_df.dropna()
		top_df['title_sentiment'] = top_df['title'].apply(lambda x: data_init.find_sentiment(x))
		top_df['description_sentiment'] = top_df['description'].apply(lambda x: data_init.find_sentiment(x))
		top_df['toi_description_score'] = top_df['description'].apply(lambda x: data_init.find_score(x))
		top_df['toi_title_score'] = top_df['title'].apply(lambda x: data_init.find_score(x))
		score_df['toi_description_score']=top_df['toi_description_score']
		score_df['toi_title_score']=top_df['toi_title_score']  
		
		


		x=0
		while x < count:
			container=st.container()
			with container:
				container2 =container.container()
				with container2:
					col1 , col2 , col3= container2.columns((0.1,2,0.6))
					
					col1.write(index)
					col2.write(top_df['title'][x])
					col2.write("")
					
					if top_df['title_sentiment'][x] != "Neutral":
						col3.metric("",top_df['title_sentiment'][x],float(top_df['toi_title_score'][x]))
						col3.write("")
						col3.write("")
					col2.markdown(top_df['description'][x])
					col2.write("")
					if top_df['description_sentiment'][x]!= "Neutral":
						col3.metric("",top_df['description_sentiment'][x],float(top_df['toi_description_score'][x]))
						col3.write("")
						col3.write("")
			x=x+1
			index=index+1


	my_expander = st.expander(label='Recent BBC feeds with sentiment and polarity scores')
	with my_expander:
		#BBC
		url = "http://feeds.bbci.co.uk/news/rss.xml"

		count=5
		index=1
		# top_df =pd.DataFrame()
		top_df = data_init.get_rss_feed(url)
		top_df = top_df.drop(['pubDate', 'guid','link'], axis=1)
		top_df.dropna()
		top_df['title_sentiment'] = top_df['title'].apply(lambda x: data_init.find_sentiment(x))
		top_df['description_sentiment'] = top_df['description'].apply(lambda x: data_init.find_sentiment(x))
		top_df['bbc_description_score'] = top_df['description'].apply(lambda x: data_init.find_score(x))
		top_df['bbc_title_score'] = top_df['title'].apply(lambda x: data_init.find_score(x))
		score_df['bbc_description_score']=top_df['bbc_description_score']
		score_df['bbc_title_score']=top_df['bbc_title_score']  
		


		x=0
		while x < count:
			container=st.container()
			with container:
				container2 =container.container()
				with container2:
					col1 , col2 , col3= container2.columns((0.1,2,0.6))
					
					col1.write(index)
					col2.write(top_df['title'][x])
					col2.write("")
					
					if top_df['title_sentiment'][x] != "Neutral":
						col3.metric("",top_df['title_sentiment'][x],float(top_df['bbc_title_score'][x]))
						col3.write("")
						col3.write("")
					col2.markdown(top_df['description'][x])
					col2.write("")
					if top_df['description_sentiment'][x]!= "Neutral":
						col3.metric("",top_df['description_sentiment'][x],float(top_df['bbc_description_score'][x]))
						col3.write("")
						col3.write("")
			x=x+1
			index=index+1



	my_expander = st.expander(label='Recent NYT feeds with sentiment and polarity scores')
	with my_expander:
		#TOI
		url = "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"

		count=5
		index=1
		# top_df =pd.DataFrame()
		top_df = data_init.get_rss_feed(url)
		top_df = top_df.drop(['pubDate', 'guid','link'], axis=1)
		top_df.dropna()
		top_df['title_sentiment'] = top_df['title'].apply(lambda x: data_init.find_sentiment(x))
		top_df['description_sentiment'] = top_df['description'].apply(lambda x: data_init.find_sentiment(x))
		top_df['ny_description_score'] = top_df['description'].apply(lambda x: data_init.find_score(x))
		top_df['ny_title_score'] = top_df['title'].apply(lambda x: data_init.find_score(x))
		score_df['ny_description_score']=top_df['ny_description_score']
		score_df['ny_title_score']=top_df['ny_title_score']  
		
		
		
	

		x=0
		while x < count:
			container=st.container()
			with container:
				container2 =container.container()
				with container2:
					col1 , col2 , col3= container2.columns((0.1,2,0.6))
					
					col1.write(index)
					col2.write(top_df['title'][x])
					col2.write("")
					
					if top_df['title_sentiment'][x] != "Neutral":
						col3.metric("",top_df['title_sentiment'][x],float(top_df['ny_title_score'][x]))
						col3.write("")
						col3.write("")
					col2.markdown(top_df['description'][x])
					col2.write("")
					if top_df['description_sentiment'][x]!= "Neutral":
						col3.metric("",top_df['description_sentiment'][x],float(top_df['ny_description_score'][x]))
						col3.write("")
						col3.write("")
			x=x+1
			index=index+1

	
	
	score_df.to_csv("scores_news.csv", index=False)
	sdata_df = pd.read_csv("scores_news.csv")
	tdata=sdata_df[["ny_title_score","bbc_title_score","toi_title_score"]].copy()
	ddata=sdata_df[['ny_description_score',"bbc_description_score",'toi_description_score']]
	c1,c2=st.columns((3,1))
	c1.line_chart(tdata)
	c1.line_chart(ddata)
	c1.area_chart(sdata_df)
	ny=sdata_df["ny_title_score"].mean()+sdata_df["ny_description_score"].mean()
	toi=sdata_df["toi_title_score"].mean()+sdata_df["toi_description_score"].mean()
	bbc=sdata_df["bbc_title_score"].mean()+sdata_df["bbc_description_score"].mean()
	c2.metric("Average NY Times Trend","",ny)
	c2.metric("Average TOI Trend","",toi)
	c2.metric("Average BBC News Trend","",bbc)








# hide app menu and footer custiomization
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# st.markdown(css.footer,unsafe_allow_html=True)


if __name__ == "__main__":
	main()





