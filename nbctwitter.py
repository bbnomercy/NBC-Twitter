import pandas as pd
import string
import numpy as np
import nltk
import re
import random
from nltk.tokenize import word_tokenize
from nltk import FreqDist,classify, NaiveBayesClassifier
from sklearn.metrics import confusion_matrix,classification_report
from collections import Counter

def get_all_words(cleaned_token_list):
	for tokens in cleaned_token_list:
		for token in tokens:
			yield token

def get_tweets_for_model(cleaned_tokens_list):
	for tweet_tokens in cleaned_tokens_list:
		yield dict([token, True] for token in tweet_tokens)

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
	return text

def tokenize(text):
	#disini diisi dengan stop words
	ignore_words = ['by', 'yang', 'ya', 'saya', 'dia', 'ia', 'ke', 'pun', 'rt']
	words = text.split();
	words = [w for w in words if w not in ignore_words]
	return words

if __name__ == '__main__':

	data_train_true = pd.read_excel('twitter-prostitute.xlsx')
	data_train_false = pd.read_excel('twitter-not-prostitute.xlsx')
	data_test = pd.read_csv('random-tweet-06052020.csv')
	stopwords_file = open("stopwords-id.txt", "r").readlines()

	print(f"Jumlah data training (True)\t:\t{len(data_train_true)}")
	print(f"Jumlah data training (False)\t:\t{len(data_train_false)}")
	print(f"Jumlah data test\t\t:\t{len(data_test)}")

	prostitute_tweets = data_train_true['tweet']
	not_prostitute_tweets = data_train_false['tweet']
	just_tweets = data_test['tweet']

	positive_tweet_tokens = []
	for i in prostitute_tweets:
		positive_tweet_tokens.append(tokenize(cleaning(i)))

	negative_tweet_tokens = []
	for i in not_prostitute_tweets:
		negative_tweet_tokens.append(tokenize(cleaning(i)))

	all_pos_words = get_all_words(positive_tweet_tokens)
	freq_dist_pos = FreqDist(all_pos_words)
	#print(f"Most common words in the tweets data are : {freq_dist_pos.most_common(5000)}")
	print()

	positive_tokens_for_model = get_tweets_for_model(positive_tweet_tokens)
	negative_tokens_for_model = get_tweets_for_model(negative_tweet_tokens)

	positive_dataset = [(tweet_dict, "True")
						for tweet_dict in positive_tokens_for_model]
	negative_dataset = [(tweet_dict, "False")
						for tweet_dict in negative_tokens_for_model]
	dataset = positive_dataset + negative_dataset

	random.shuffle(dataset)

	train_data = dataset[:40000]
	test_data = dataset[9600:]

	classifier = NaiveBayesClassifier.train(train_data)
	print("Akurasi Klasifikasi Naive Bayes\t:\t"+"{:.2f}".format(classify.accuracy(classifier, test_data) * 100)+" %")
	print()

	''' Pengujian akurasi dan confusion matrix '''
	test_result = []
	classifier_result = []

	for i in range(len(test_data)):
		test_result.append(classifier.classify(test_data[i][0]))
		classifier_result.append(test_data[i][1])

	c_matrix = nltk.ConfusionMatrix(classifier_result, test_result)

	print(f"Confusion Matrix :\n{c_matrix}", )

	labels = {'True', 'False'}

	TP, FN, FP = Counter(), Counter(), Counter()
	for i in labels:
		for j in labels:
			if i == j:
				TP[i] += int(c_matrix[i,j])
			else:
				FN[i] += int(c_matrix[i,j])
				FP[j] += int(c_matrix[i,j])

	print("label   | precision             | recall                | f_measure         ")
	print("--------+-----------------------+-----------------------+-------------------")
	for label in sorted(labels):
		precision, recall = 0, 0
		if TP[label] == 0:
			f_measure = 0
		else:
			precision = float(TP[label]) / (TP[label]+FP[label])
			recall = float(TP[label]) / (TP[label]+FN[label])
			f_measure = float(2) * (precision * recall) / (precision + recall)
		print(f"{label}\t| {precision}\t| {recall}\t| {f_measure}")

	# print()
	# print(classifier.show_most_informative_features(10))
	# print()

	# custom_tweet = "open follow twitter bo untuk kawan kawan semuanya"
	# cleaned_custom_tokens = tokenize(cleaning(custom_tweet))
	# result = classifier.classify(dict([token, True] for token in cleaned_custom_tokens))

	# print(f"Sample tweet: {custom_tweet}")
	# print(f"Hasil klasifikasi twitter: {result}")

	# custom_tweet = pd.read_csv('new-data-b.csv');
	# custom = custom_tweet['tweet']
	# a = 0
	# for i in custom:
	# 	a += 1
	# 	cleaned_custom_tokens = tokenize(cleaning(i))
	# 	result = classifier.classify(dict([token, True] for token in cleaned_custom_tokens))

	# 	print(f"{a} : {result}")