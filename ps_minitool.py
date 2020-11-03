import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
import string
from nltk.stem.snowball import SnowballStemmer
import warnings
import base64
from io import BytesIO
from nltk.corpus import stopwords
import unicodedata


# Função para ignorar avisos
def ignore_warn(*args, **kwargs):
	pass
warnings.warn = ignore_warn

# Imagem
img = Image. open('logo_ps.png')
st.image(img)

# Título
st.title('MiniTool - Pulse Solution')

# Função para a primeira limpeza de texto
def clean_text_round1(text):
    # Transforma o texto em letra minúscula, retira pontuações, aspas, parênteses e palavras com números
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text


# Função para a segunda limpeza de texto
def stem_sentences(sentence):
    # Deixa apenas a raiz das palavras
    portugueseStemmer = SnowballStemmer("portuguese", ignore_stopwords=True)
    tokens = sentence.split()
    stemmed_tokens = [portugueseStemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

# Função para transformar df em excel
def to_excel(df):
	output = BytesIO()
	writer = pd.ExcelWriter(output, engine='xlsxwriter')
	df.to_excel(writer, sheet_name='Planilha1',index=False)
	writer.save()
	processed_data = output.getvalue()
	return processed_data
	
# Função para gerar link de download
def get_table_download_link(df):
	val = to_excel(df)
	b64 = base64.b64encode(val)
	return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download</a>'

# Carregando modelo de CATEGORIZAÇÃO
with open('model_cat_1809', 'rb') as f:
    model_cat = pickle.load(f)


# Função para fazer a predição de CATEGORIZAÇÃO
def make_prediction_cat(text):

	text = pd.Series(text)
	text = text.apply(clean_text_round1)
	text = text.apply(stem_sentences)
	result = model_cat.predict(text) # Aplicando modelo
	return result

# Carregando modelo de SENTIMENTAÇÃO
with open('model_sent_1809', 'rb') as f:
	model_sent = pickle.load(f)

# Função para fazer a predição de SENTIMENTAÇÃO
def make_prediction_sent(text):

	text = pd.Series(text)
	text = text.apply(clean_text_round1)
	text = text.apply(stem_sentences)
	result = model_sent.predict(text) # Aplicando modelo
	return result

# Carregando modelo de Subcategorização de Elogio
with open('model_sub_elo', 'rb') as f:
	model_sub_elo = pickle.load(f)

# Função para fazer a predição de Subcategorização de Elogio
def make_prediction_sub_elo(text):

    text = pd.Series(text)
    text = text.apply(clean_text_round1)
    text = text.apply(stem_sentences)
    result = model_sub_elo.predict(text)
    return result

# Carregando modelo de Subcategorização de Reclamação - Finanças
with open('model_fin', 'rb') as f:
	model_sub_rec_fin = pickle.load(f)

# Função para fazer a predição de Subcategorização de Reclamação - Finanças
def make_prediction_sub_rec_fin(text):

    text = pd.Series(text)
    text = text.apply(clean_text_round1)
    text = text.apply(stem_sentences)
    result = model_sub_rec_fin.predict(text)
    return result

# Carregando modelo de Subcategorização de Reclamação - Delivery
with open('model_sub_rec_del', 'rb') as f:
	model_sub_rec_del = pickle.load(f)

# Função para fazer a predição de Subcategorização de Reclamação - Delivery
def make_prediction_sub_rec_del(text):

	text = pd.Series(text)
	text = text.apply(clean_text_round1)
	text = text.apply(stem_sentences)
	result = model_sub_rec_del.predict(text)
	return result

# Carregando modelo de Subcategorização de Reclamação - Compras
with open('model_com', 'rb') as f:
	model_sub_rec_com = pickle.load(f)

# Função para fazer a predição de Subcategorização de Reclamação - Compras
def make_prediction_sub_rec_com(text):

	text = pd.Series(text)
	text = text.apply(clean_text_round1)
	text = text.apply(stem_sentences)
	result = model_sub_rec_com.predict(text)
	return result

# Carregando modelo de Subcategorização de Reclamação - Claro
with open('model_sub_rec_claro', 'rb') as f:
	model_sub_rec_claro = pickle.load(f)

# Função para fazer a predição de Subcategorização de Reclamação - Claro
def make_prediction_sub_rec_claro(text):

	text = pd.Series(text)
	text = text.apply(clean_text_round1)
	text = text.apply(stem_sentences)
	result = model_sub_rec_claro.predict(text)
	return result

# Função para remover acentos
def strip_accents(text):

    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)

def clean_freq_word(txt):
	txt = txt.lower()
	txt = strip_accents(txt)
	txt = re.sub('[%s]' % re.escape(punctuation), '', txt)
	return str(txt)

# Listas de possibilidades de data e reviews
nomes_reviews = ('text','Text','TEXT','comentario','Comentario','COMENTARIO','comentário','Comentário','COMENTÁRIO',
	'Comentarios','COMENTARIOS','comentários','Comentários','COMENTÁRIOS','texto','Texto','TEXTO',
	'textos','Textos','TEXTOS','body','Body','BODY','review','Review','REVIEW','reviews','Reviews','REVIEWS')
nomes_datas = ('date','Date','DATE','dates','Dates','DATES','data','Data','DATA','datas','Datas','DATAS','dia','Dia','DIA',
	'dias','Dias','DIAS','mes','Mes','MES','mês','Mês','MÊS','day','Day','DAYS')


# Menu
menu = ['Finanças', 'Delivery', 'Compras', 'Claro', 'Outros']
choices = st.sidebar.selectbox('Selecione a categoria', menu)

if choices == 'Finanças':


	# Header
	st.header('Classificação de Reviews - Finanças')

	####### Categorização e sentimentação de frases #######
	st.subheader('Escreva uma frase para verificar seu sentimento e categoria')

	frase = st.text_input('')
	frase = pd.Series(frase)

	st.write('O sentimento desta frase é : ', make_prediction_sent(frase)[0])
	st.write('A categoria desta frase é : ', make_prediction_cat(frase)[0])

	####### Upload dataset #######
	st.subheader('MioGinho')
	data = st.file_uploader("Insira a base de dados", type='xlsx')
	if data is not None:
		df = pd.read_excel(data)
		st.write(df.head(10))
		st.write('Este arquivo possui ', len(df.index), ' linhas e ', len(df.columns), ' colunas.')

	# Apagando coluna Review ID
	if 'Review ID' in df.columns:
		df.drop('Review ID', axis=1, inplace=True)

	# Fazendo as alterações nas colunas de data e review
	df.columns = ['Review' if c.startswith(nomes_reviews) else c for c in df.columns]
	df.columns = ['Data' if c.startswith(nomes_datas) else c for c in df.columns]

	# Transformando int em str
	df['Review'] = [str (item) for item in df['Review']]
	df['Review'] = [item for item in df['Review'] if not isinstance(item, int)]

	
	####### Contador de palavras específicas #######
	st.subheader('Contador de palavras')

	palavra = st.text_input('Insira uma palavra para verificar o número de ocorrências diárias ')
	palavra = palavra.lower()
	palavra = strip_accents(palavra)

	# Contando a palavra definida e criando um dataframe
	palavra_cont = df[['Data', 'Review']]
	palavra_cont['Review'] = palavra_cont['Review'].str.lower()
	palavra_cont['Review'] = palavra_cont['Review'].apply(strip_accents)
	palavra_cont['Quantidade'] = palavra_cont['Review'].str.count(palavra)
	palavra_cont = palavra_cont[['Data', 'Quantidade']]
	contagem = (palavra_cont['Quantidade'].gt(0)).groupby(palavra_cont['Data']).sum().to_frame(palavra).reset_index()
	contagem[palavra] = contagem[palavra].astype(int)

	# Apresentando os resultados
	st.write(contagem.head(10))
	st.write('Existem ', contagem[palavra].sum(), 'ocorrências da palavra ', palavra, 'na base de dados')

	# Download da tabela
	st.markdown(get_table_download_link(contagem), unsafe_allow_html=True)


	####### Contador de palavras comuns #######
	st.subheader('Palavras mais frequentes')

	# Inserindo top N palavras
	top_n = st.number_input('Digite o número de palavras mais frequentes que deseja visualizar:', value=10)

	stop_words = ['de','a','o','que','e','é','do','da','em','um','para','com','não','uma','os','no','se','na','por','mais','as','dos','como',
		'mas','ao','ele','das','à','seu','sua','ou','quando','muito','nos','já','eu','também','só','pelo','pela','até','isso','ela','entre','depois',
		'sem','mesmo','aos','seus','quem','nas','me','esse','eles','você','essa','num','nem','suas','meu','às','minha','numa','pelos','elas','qual',
		'nós','lhe','deles','essas','esses','pelas','este','dele','tu','te','vocês','vos','lhes','meus','minhas','teu','tua','teus','tuas','nosso',
		'nossa','nossos','nossas','dela','delas','esta','estes','estas','aquele','aquela','aqueles','aquelas','isto','aquilo','estou','está','estamos',
		'estão','estive','esteve','estivemos','estiveram','estava','estávamos','estavam','estivera','estivéramos','esteja','estejamos','estejam','estivesse',
		'estivéssemos','estivessem','estiver','estivermos','estiverem','hei','há','havemos','hão','houve','houvemos','houveram','houvera','houvéramos',
		'haja','hajamos','hajam','houvesse','houvéssemos','houvessem','houver','houvermos','houverem','houverei', 'houverá','houveremos','houverão',
		'houveria','houveríamos','houveriam','sou','somos','são','era', 'éramos','eram','fui','foi','fomos','foram','fora','fôramos','seja','sejamos',
		'sejam','fosse','fôssemos','fossem','for','formos','forem','serei','será','seremos','serão','seria','seríamos','seriam','tenho','tem','temos','tém',
		'tinha','tínhamos','tinham','tive','teve','tivemos','tiveram','tivera','tivéramos','tenha','tenhamos','tenham','tivesse','tivéssemos','tivessem',
		'tiver','tivermos','tiverem','terei','terá','teremos','terão','teria','teríamos','teriam','pra']
	punctuation = '!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

	# Aplicando limpeza de texto p/ palavras mais frequentes
	txt = df['Review']
	txt = txt.apply(clean_freq_word)

	# Separando as palavras em tokens, retirando stopwords e criando dataframe
	txt = txt.str.cat(sep=' ')
	words_txt = nltk.tokenize.word_tokenize(txt)
	words_dist = nltk.FreqDist(words_txt)
	words_dist_stop = nltk.FreqDist(w for w in words_txt if w not in stop_words)
	words_common = pd.DataFrame(words_dist_stop.most_common(top_n), columns=['Palavra', 'Quantidade'])
	
	# Exibindo resultado
	st.write(words_common)

	# Download da tabela
	st.markdown(get_table_download_link(words_common), unsafe_allow_html=True)


	####### Fazendo previsões #######
	st.subheader('Clique em Predict para obter as classificações da sua base de dados')
	st.write('')

	btn_predict = st.button('Predict')
	if btn_predict:

		reviews = df['Review']

		# Aplicando modelo de Categorização e Sentimentação nos reviews
		reviews_cat = make_prediction_cat(reviews)
		reviews_sent = make_prediction_sent(reviews)

		# Criando dataframe com a data, review, categorização e sentimentação
		reviews_done = pd.DataFrame({'Data': df['Data'],
			'Review': df['Review'],
			'Sentimentação': reviews_sent,
			'Categorização': reviews_cat})

		reviews_done['Subcategorização'] = np.where(reviews_done['Categorização']=='Elogio',
		 make_prediction_sub_elo(reviews_done['Review']), make_prediction_sub_rec_fin(reviews_done['Review']))

		st.write(reviews_done.head(10))
		st.write('''Para salvar, clique em Download ou clique com o botão direito, selecione Salvar link como... 
			e nomeie o arquivo com a extensão .xlsx''')
		st.markdown(get_table_download_link(reviews_done), unsafe_allow_html=True)


if choices == 'Delivery':


	# Header
	st.header('Categorização de Reviews - Delivery')

	####### Categorização e sentimentação de frases #######
	st.subheader('Escreva uma frase para verificar seu sentimento e categoria')

	frase = st.text_input('')
	frase = pd.Series(frase)

	st.write('O sentimento desta frase é ', make_prediction_sent(frase)[0])
	st.write('A categoria desta frase é ', make_prediction_cat(frase)[0])

	####### Upload dataset #######
	st.write("MioGinho")

	data = st.file_uploader("Insira a base de dados", type='xlsx')
	if data is not None:
		df = pd.read_excel(data)
		st.write(df.head(10))
		st.write('Este arquivo possui ', len(df.index), ' linhas e ', len(df.columns), ' colunas.')

	# Apagando coluna Review ID
	if 'Review ID' in df.columns:
		df.drop('Review ID', axis=1, inplace=True)

	# Fazendo as alterações nas colunas de data e review
	df.columns = ['Review' if c.startswith(nomes_reviews) else c for c in df.columns]
	df.columns = ['Data' if c.startswith(nomes_datas) else c for c in df.columns]

	# Transformando int em str
	df['Review'] = [str (item) for item in df['Review']]
	df['Review'] = [item for item in df['Review'] if not isinstance(item, int)]

	
	####### Contador de palavras específicas #######
	st.subheader('Contador de palavras')

	palavra = st.text_input('Insira uma palavra para verificar o número de ocorrências diárias: ')
	palavra = palavra.lower()
	palavra = strip_accents(palavra)

	# Contando a palavra definida e criando um dataframe
	palavra_cont = df[['Data', 'Review']]
	palavra_cont['Review'] = palavra_cont['Review'].str.lower()
	palavra_cont['Review'] = palavra_cont['Review'].apply(strip_accents)
	palavra_cont['Quantidade'] = palavra_cont['Review'].str.count(palavra)
	palavra_cont = palavra_cont[['Data', 'Quantidade']]
	contagem = (palavra_cont['Quantidade'].gt(0)).groupby(palavra_cont['Data']).sum().to_frame(palavra).reset_index()
	contagem[palavra] = contagem[palavra].astype(int)

	# Apresentando os resultados
	st.write(contagem.head(10))
	st.write('Existem ', contagem[palavra].sum(), 'ocorrências da palavra ', palavra, 'na base de dados')

	# Download da tabela
	st.markdown(get_table_download_link(contagem), unsafe_allow_html=True)


	####### Contador de palavras comuns #######
	st.subheader('Palavras mais frequentes')

	# Inserindo top N palavras
	top_n = st.number_input('Digite o número de palavras mais frequentes que deseja visualizar:', value=10)

	stop_words = stopwords.words('portuguese')
	punctuation = '!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

	# Aplicando limpeza de texto p/ palavras mais frequentes
	txt = df['Review']
	txt = txt.apply(clean_freq_word)

	# Separando as palavras em tokens, retirando stopwords e criando dataframe
	txt = txt.str.cat(sep=' ')
	words_txt = nltk.tokenize.word_tokenize(txt)
	words_dist = nltk.FreqDist(words_txt)
	words_dist_stop = nltk.FreqDist(w for w in words_txt if w not in stop_words)
	words_common = pd.DataFrame(words_dist_stop.most_common(top_n), columns=['Palavra', 'Quantidade'])
	
	# Exibindo resultado
	st.write(words_common)

	# Download da tabela
	st.markdown(get_table_download_link(words_common), unsafe_allow_html=True)


	####### Fazendo previsões #######
	st.subheader('Clique em Predict para obter as classificações da sua base de dados')
	st.write('')

	btn_predict = st.button('Predict')
	if btn_predict:

		reviews = df['Review']

		# Aplicando modelo de Categorização e Sentimentação nos reviews
		reviews_cat = make_prediction_cat(reviews)
		reviews_sent = make_prediction_sent(reviews)

		# Criando dataframe com a data, review, categorização e sentimentação
		reviews_done = pd.DataFrame({'Data': df['Data'],
			'Review': df['Review'],
			'Sentimentação': reviews_sent,
			'Categorização': reviews_cat})

		reviews_done['Subcategorização'] = np.where(reviews_done['Categorização']=='Elogio',
		 make_prediction_sub_elo(reviews_done['Review']), make_prediction_sub_rec_del(reviews_done['Review']))

		st.write(reviews_done.head(10))
		st.write('''Para salvar, clique em Download ou clique com o botão direito, selecione Salvar link como... 
			e nomeie o arquivo com a extensão .xlsx''')
		st.markdown(get_table_download_link(reviews_done), unsafe_allow_html=True)

if choices == 'Compras':

	# Header
	st.header('Categorização de Reviews - Compras')

	####### Categorização e sentimentação de frases #######
	st.subheader('Escreva uma frase para verificar seu sentimento e categoria')


	frase = st.text_input('')
	frase = pd.Series(frase)

	st.write('O sentimento desta frase é ', make_prediction_sent(frase)[0])
	st.write('A categoria desta frase é ', make_prediction_cat(frase)[0])

	####### Upload dataset #######
	st.subheader('MioGinho')
	data = st.file_uploader("Insira a base de dados", type='xlsx')
	if data is not None:
		df = pd.read_excel(data)
		st.write(df.head(10))
		st.write('Este arquivo possui ', len(df.index), ' linhas e ', len(df.columns), ' colunas.')

	# Apagando coluna Review ID
	if 'Review ID' in df.columns:
		df.drop('Review ID', axis=1, inplace=True)

	# Fazendo as alterações nas colunas de data e review
	df.columns = ['Review' if c.startswith(nomes_reviews) else c for c in df.columns]
	df.columns = ['Data' if c.startswith(nomes_datas) else c for c in df.columns]

	# Transformando int em str
	df['Review'] = [str (item) for item in df['Review']]
	df['Review'] = [item for item in df['Review'] if not isinstance(item, int)]

	
	####### Contador de palavras específicas #######
	st.subheader('Contador de palavras')

	palavra = st.text_input('Insira uma palavra para verificar o número de ocorrências diárias: ')
	palavra = palavra.lower()
	palavra = strip_accents(palavra)

	# Contando a palavra definida e criando um dataframe
	palavra_cont = df[['Data', 'Review']]
	palavra_cont['Review'] = palavra_cont['Review'].str.lower()
	palavra_cont['Review'] = palavra_cont['Review'].apply(strip_accents)
	palavra_cont['Quantidade'] = palavra_cont['Review'].str.count(palavra)
	palavra_cont = palavra_cont[['Data', 'Quantidade']]
	contagem = (palavra_cont['Quantidade'].gt(0)).groupby(palavra_cont['Data']).sum().to_frame(palavra).reset_index()
	contagem[palavra] = contagem[palavra].astype(int)

	# Apresentando os resultados
	st.write(contagem.head(10))
	st.write('Existem ', contagem[palavra].sum(), 'ocorrências da palavra ', palavra, 'na base de dados')

	# Download da tabela
	st.markdown(get_table_download_link(contagem), unsafe_allow_html=True)


	####### Contador de palavras comuns #######
	st.subheader('Palavras mais frequentes')

	# Inserindo top N palavras
	top_n = st.number_input('Digite o número de palavras mais frequentes que deseja visualizar:', value=10)

	stop_words = stopwords.words('portuguese')
	punctuation = '!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

	# Aplicando limpeza de texto p/ palavras mais frequentes
	txt = df['Review']
	txt = txt.apply(clean_freq_word)

	# Separando as palavras em tokens, retirando stopwords e criando dataframe
	txt = txt.str.cat(sep=' ')
	words_txt = nltk.tokenize.word_tokenize(txt)
	words_dist = nltk.FreqDist(words_txt)
	words_dist_stop = nltk.FreqDist(w for w in words_txt if w not in stop_words)
	words_common = pd.DataFrame(words_dist_stop.most_common(top_n), columns=['Palavra', 'Quantidade'])
	
	# Exibindo resultado
	st.write(words_common)

	# Download da tabela
	st.markdown(get_table_download_link(words_common), unsafe_allow_html=True)


	####### Fazendo previsões #######
	st.subheader('Clique em Predict para obter as classificações da sua base de dados')
	st.write('')

	btn_predict = st.button('Predict')
	if btn_predict:

		reviews = df['Review']

		# Aplicando modelo de Categorização e Sentimentação nos reviews
		reviews_cat = make_prediction_cat(reviews)
		reviews_sent = make_prediction_sent(reviews)

		# Criando dataframe com a data, review, categorização e sentimentação
		reviews_done = pd.DataFrame({'Data': df['Data'],
			'Review': df['Review'],
			'Sentimentação': reviews_sent,
			'Categorização': reviews_cat})

		reviews_done['Subcategorização'] = np.where(reviews_done['Categorização']=='Elogio',
			make_prediction_sub_elo(reviews_done['Review']), make_prediction_sub_rec_com(reviews_done['Review']))

		st.write(reviews_done.head(10))
		st.write('''Para salvar, clique em Download ou clique com o botão direito, selecione Salvar link como... 
			e nomeie o arquivo com a extensão .xlsx''')
		st.markdown(get_table_download_link(reviews_done), unsafe_allow_html=True)

if choices == 'Claro':

	# Header
	st.header('Categorização de Reviews - Claro')

	####### Categorização e sentimentação de frases #######
	st.subheader('Escreva uma frase para verificar seu sentimento e categoria')

	frase = st.text_input('')
	frase = pd.Series(frase)

	st.write('O sentimento desta frase é ', make_prediction_sent(frase)[0])
	st.write('A categoria desta frase é ', make_prediction_cat(frase)[0])

	####### Upload dataset #######
	st.subheader('MioGinho')
	data = st.file_uploader("Insira a base de dados", type='xlsx')
	if data is not None:
		df = pd.read_excel(data)
		st.write(df.head(10))
		st.write('Este arquivo possui ', len(df.index), ' linhas e ', len(df.columns), ' colunas.')

	# Apagando coluna Review ID
	if 'Review ID' in df.columns:
		df.drop('Review ID', axis=1, inplace=True)

	# Fazendo as alterações nas colunas de data e review
	df.columns = ['Review' if c.startswith(nomes_reviews) else c for c in df.columns]
	df.columns = ['Data' if c.startswith(nomes_datas) else c for c in df.columns]

	# Transformando int em str
	df['Review'] = [str (item) for item in df['Review']]
	df['Review'] = [item for item in df['Review'] if not isinstance(item, int)]

	
	####### Contador de palavras específicas #######
	st.subheader('Contador de palavras')

	palavra = st.text_input('Insira uma palavra para verificar o número de ocorrências diárias: ')
	palavra = palavra.lower()
	palavra = strip_accents(palavra)

	# Contando a palavra definida e criando um dataframe
	palavra_cont = df[['Data', 'Review']]
	palavra_cont['Review'] = palavra_cont['Review'].str.lower()
	palavra_cont['Review'] = palavra_cont['Review'].apply(strip_accents)
	palavra_cont['Quantidade'] = palavra_cont['Review'].str.count(palavra)
	palavra_cont = palavra_cont[['Data', 'Quantidade']]
	contagem = (palavra_cont['Quantidade'].gt(0)).groupby(palavra_cont['Data']).sum().to_frame(palavra).reset_index()
	contagem[palavra] = contagem[palavra].astype(int)

	# Apresentando os resultados
	st.write(contagem.head(10))
	st.write('Existem ', contagem[palavra].sum(), 'ocorrências da palavra ', palavra, 'na base de dados')

	# Download da tabela
	st.markdown(get_table_download_link(contagem), unsafe_allow_html=True)


	####### Contador de palavras comuns #######
	st.subheader('Palavras mais frequentes')

	# Inserindo top N palavras
	top_n = st.number_input('Digite o número de palavras mais frequentes que deseja visualizar:', value=10)

	stop_words = stopwords.words('portuguese')
	punctuation = '!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

	# Aplicando limpeza de texto p/ palavras mais frequentes
	txt = df['Review']
	txt = txt.apply(clean_freq_word)

	# Separando as palavras em tokens, retirando stopwords e criando dataframe
	txt = txt.str.cat(sep=' ')
	words_txt = nltk.tokenize.word_tokenize(txt)
	words_dist = nltk.FreqDist(words_txt)
	words_dist_stop = nltk.FreqDist(w for w in words_txt if w not in stop_words)
	words_common = pd.DataFrame(words_dist_stop.most_common(top_n), columns=['Palavra', 'Quantidade'])
	
	# Exibindo resultado
	st.write(words_common)

	# Download da tabela
	st.markdown(get_table_download_link(words_common), unsafe_allow_html=True)


	####### Fazendo previsões #######
	st.subheader('Clique em Predict para obter as classificações da sua base de dados')
	st.write('')

	btn_predict = st.button('Predict')
	if btn_predict:

		reviews = df['Review']

		# Aplicando modelo de Categorização e Sentimentação nos reviews
		reviews_cat = make_prediction_cat(reviews)
		reviews_sent = make_prediction_sent(reviews)

		# Criando dataframe com a data, review, categorização e sentimentação
		reviews_done = pd.DataFrame({'Data': df['Data'],
			'Review': df['Review'],
			'Sentimentação': reviews_sent,
			'Categorização': reviews_cat})

		reviews_done['Subcategorização'] = np.where(reviews_done['Categorização']=='Elogio',
		 make_prediction_sub_elo(reviews_done['Review']), make_prediction_sub_rec_claro(reviews_done['Review']))

		st.write(reviews_done.head(10))
		st.write('''Para salvar, clique em Download ou clique com o botão direito, selecione Salvar link como... 
			e nomeie o arquivo com a extensão .xlsx''')
		st.markdown(get_table_download_link(reviews_done), unsafe_allow_html=True)

if choices == 'Outros':


 		# Header
	st.header('Categorização de Reviews - Outros')
