import pandas as pd
import string
import numpy as np
import re
import random
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.naive_bayes import MultinomialNB

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

class FasterStemmer(object):
	def __init__(self):
		self.words = {}

	def stem(self, x):
		if x in self.words:
			return self.words[x]
		t = stemmer.stem(x)
		self.words[x] = t
		return t

fast_stemmer = FasterStemmer()

stopwords_file = open("stopwords-id.txt", 'r')
stopwords = [x.strip() for x in stopwords_file.readlines()]
stopwords.extend(['by', 'rt', 'via'])

def cleaning(text):
	text = re.sub(r'<[^>]+>', '', text) #delete html tags
	text = re.sub(r'\S*twitter.com\S*', '', text)   #delete twitter image
	text = re.sub(r'https?://[A-Za-z0-9./]+','',text) #delete url
	text = re.sub(r'@[A-Za-z0-9]+','',text) #delete user mention
	text = re.sub(r'#[A-Za-z0-9]+','',text) #delete twitter hashtag
	text = re.sub(r'(?:(?:\d+,?)+(?:\.?\d+)?)','', text) #delete number
	text = re.sub(r"[^a-zA-Z]", " ", text) #only accept alphabet char
	text = re.sub(r"(\w)(\1{2,})", r'\1', text) #delete repeated char
	text = re.sub(r"\b[a-zA-Z]\b", "", text) #remove single character
	text = text.lower() #change to lowercase
	text = fast_stemmer.stem(text) #stemming
	return text

def tokenize(text):
	words = text.split();
	words = [w for w in words if w not in stopwords]
	return words

if __name__ == '__main__':

	print('Loading dataset...\n')
	data_train_true = pd.read_excel('twitter-prostitute.xlsx')
	data_train_false = pd.read_excel('twitter-not-prostitute.xlsx')
	dataset = pd.concat([data_train_true, data_train_false], ignore_index = True)

	dataset.to_excel("dataset.xlsx",index = False)

	#dataset = pd.read_excel('labeled-data-testing.xlsx')

	print(f'{len(dataset)} Total dataset\n')

	print('Preprocessing...\n')
	dataset['label'] = dataset.status.map(lambda x: x)
	dataset['tweet'] = dataset.tweet.map(lambda x: cleaning(x))
	dataset['tweet'] = dataset.tweet.apply(lambda x: tokenize(x))
	dataset['tweet'] = dataset.tweet.apply(lambda x: ' '.join(x))

	print('Building the model...\n')
	count_vect = CountVectorizer()
	counts = count_vect.fit_transform(dataset['tweet'])

	transformer = TfidfTransformer().fit(counts)
	counts = transformer.transform(counts)

	feature_train, feature_test, target_train, target_test = train_test_split(counts, dataset['label'], train_size=0.8, test_size=0.2, random_state=52)
	model = MultinomialNB()
	model.fit(feature_train, target_train)
	predicted = model.predict(feature_test)

	accuracy = accuracy_score(target_test, predicted)
	c_matrix = pd.DataFrame(
						confusion_matrix(target_test, predicted, labels=[1, 0]),
						index = ['Actual:True', 'Actual:False'],
						columns = ['Pred:True', 'Pred:False']
				);
	c_report = classification_report(target_test, predicted)

	print(f'NBC accuracy : {accuracy*100}%')
	print()
	print(c_matrix)
	print()
	print(c_report)

	input_text = ''

	while(input_text!='exit'):
		text = input('Masukkan tweet : ')
		print('\n')

		if text == 'exit':
			print('Exiting program [Bye]\n')
			input_text = 'exit'
		else:
			text = tokenize(cleaning(text))

			new_counts = count_vect.transform(text)
			print(new_counts)
			pred = model.predict(new_counts)

			if pred[0] == 0:
				print("False (Not Prostitute)\n")
			else:
				print('True (Prostitute)\n\n')

			#new_pred = model.predict_proba(new_counts)
			#print(new_pred)

