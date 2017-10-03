# ! pip3.5 install scattertext && python -m spacy.en.download
# ! pip3.5 install webhoseio
# Enter Webhose_key before proceeding!


## Takes text data from CNN and Fox News and compares the output of each using
# Natural Language Processing word frequency statistics

import webhoseio
import pandas as pd
import scattertext as st
import spacy.en
from scattertext import CorpusFromPandas, produce_scattertext_explorer, word_similarity_explorer
from IPython.display import IFrame

webhose_key = # Enter Webhose Key Here

webhoseio.config(token=webhose_key)

def get_headlines(search_term, site):
	query_params = {
		"q": search_term + " site:" + site + ".com language:english",
		"sort": "published"
	    }
	output = webhoseio.query("filterWebContent", query_params)
	print('[-] creating ' + site + '_output.txt')
	file = open(site + '_output.txt','w') 
	try:
		for x in range(100):
			file.write(output['posts'][x]['text'])
	except IndexError:
		print('[-] Warning: less than 100 results')
	file.close()
	print('[+] operation complete')

get_headlines('trump', 'cnn')
get_headlines('trump', 'foxnews')

# Read cnn_output.txt and save as pandas dataframe
cnnfile = open('cnn_output.txt', "r")
lines = cnnfile.read().split("\n")
CNN = ["" for x in range(len(lines))]
i = 0
for l in lines:
    values = l.split("\t") # use tab delimiter
    CNN[i] = values[0]
    i = i + 1
df1 = { 'CNN':CNN }
df1 = pd.DataFrame(df1)

# Read foxnews_output.txt and save as pandas dataframe
foxnewsfile = open('foxnews_output.txt', "r")
lines2 = foxnewsfile.read().split("\n")
Fox = ["" for x in range(len(lines2))]
i = 0
for l in lines2:
    values = l.split("\t") # use tab delimiter
    Fox[i] = values[0]
    i = i + 1
df2 = { 'Fox':Fox }
df2 = pd.DataFrame(df2)

# Join the two dataframes along the column
convention_df = pd.concat([df1, df2], axis=1)

# Place all text in same column and create tag for CNN or Fox
convention_df = pd.melt(convention_df)
convention_df = convention_df.dropna(axis=0, how='any')

# Build NLP parsing for corpus
English = st.whitespace_nlp_with_sentences

# Parse the text and create new column with parsed values
convention_df.groupby('variable').apply(lambda x: x.value.apply(lambda x: len(x.split())).sum())
convention_df['parsed'] = convention_df.value.apply(English)

convention_df.iloc[:3]

# Generate corpus of language from pandas dataframe
corpus = st.CorpusFromPandas(convention_df, category_col='variable', text_col='value', nlp = English).build()

# Output html doc for visualization
## HTML FILE MUST ALREADY EXIST IN OUTPUT FOLDER TO WRITE ON
html = st.produce_scattertext_explorer(corpus,
                                       category='CNN',
                                       category_name='CNN',
                                       not_category_name='Fox',
                                       width_in_pixels=1000)

file_name = 'output/Trump.html'
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width = 1200, height=700)