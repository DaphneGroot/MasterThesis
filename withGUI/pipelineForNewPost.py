import sys
# import csv
import pandas as pd
import numpy as np
import os
import shutil
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score
from sklearn.svm import SVR, LinearSVR, SVC
from sklearn.linear_model import SGDRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV
from scipy.stats import entropy
from nltk.corpus import stopwords

import math
import collections, re

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

def controversialLevel(values):
	# based on average entropy of trainingset
	if values < 0.477:
		if values < 0.2385:
			level = "low"
		else:
			level = "medium-low"
	elif values >= 0.477:
		if values < 0.7155:
			level = "medium-high"
		else:
			level = "high"


	return level


def H(values):
    # compute probabily vector
    # print(values)
    if sum(values) != 0: #to prevent 'division by 0'
        probvect = [int(x)/sum(values) for x in values]

        # computer entropy over prob vector
        h = entropy(probvect)
    else:
        h = 0

    return h


def predictEntropy(trainData, predictedReactionsVolume):
	predictedEntropy = H(predictedReactionsVolume)
	# print('Predcited entropy score: ', predictedEntropy)


	''' have to calculate entropy accuracy'''

	controversial = controversialLevel(predictedEntropy)
	# print('Predicted controversial level: ', controversial)

	return predictedEntropy, controversial


def predictReactionsVolume(trainData, title, summary, message, totalReactions):
	'''with normalized'''
	reactions = ['normalizedLIKE','normalizedLOVE','normalizedHAHA','normalizedWOW','normalizedSAD','normalizedANGRY']
	
	predictedReactionsNormalized = []
	predictedReactionsAbsolute = []

	for reaction in reactions:
		trainX = trainData["title"]+' '+trainData['summary']+' '+trainData['message']
		trainY = trainData[reaction]

		testX = [title+' '+summary+' '+message]

		classifier = Pipeline([('features',FeatureUnion([
									  ('word', TfidfVectorizer(analyzer='word',
															   ngram_range=(1,4),
															   binary=False,
															   norm='l2',
															   sublinear_tf=False)),
									  ('char', TfidfVectorizer(analyzer='char',
															   ngram_range=(1,6),
															   binary=True,
															   norm='l2',
															   sublinear_tf=True
															   ))])),
								('clf', LinearSVR(C=1.0))])

		classifier.fit(trainX,trainY)

		predictedY = classifier.predict(testX)

		predictedReactionsNormalized.append(predictedY)

	# check if negative and make all positive
	
	# 1 negative: https://www.facebook.com/paroolnl/posts/10155402090351219
	# more negatives: https://www.facebook.com/volkskrant/posts/1899857450035957
	negatives = np.array(predictedReactionsNormalized) < 0
	if True in negatives:
		for idx,i in enumerate(negatives):
			if i == True:
				numberToDivide = predictedReactionsNormalized[idx]
				subtractPerNumber = numberToDivide/list(negatives).count(False)

				for idx2, n in enumerate(negatives):
					if n == False:
						predictedReactionsNormalized[idx2] = predictedReactionsNormalized[idx2]-subtractPerNumber
					
				predictedReactionsNormalized[idx] = np.array([0.0])

	# convert to absolute numbers
	for i in predictedReactionsNormalized:
		x = i * totalReactions
		predictedYabsolute = int(round(x[0]))
		predictedReactionsAbsolute.append(predictedYabsolute)


	return predictedReactionsNormalized, predictedReactionsAbsolute





def predictTotalReactions(trainData, title, summary, message):
	trainX = trainData["title"]+' '+trainData['summary']+' '+trainData['message']
	trainY = trainData['normalizedTotalReactions']

	testX = [title+' '+summary+' '+message]



	classifier = Pipeline([('features',FeatureUnion([
									  ('word', TfidfVectorizer(analyzer='word',
															   ngram_range=(2,3),
															   binary=False,
															   norm='l2',
															   sublinear_tf=True)),
									  ('char', TfidfVectorizer(analyzer='char',
															   ngram_range=(1,5),
															   binary=True,
															   norm='l2',
															   sublinear_tf=True
															   ))])),
							('clf', LinearSVR(C=1.0))])

	classifier.fit(trainX,trainY)

	predictedY = classifier.predict(testX)

	predictedYabsoluteArray = 10**predictedY
	predictedYabsolute = int(round(predictedYabsoluteArray[0]))
	
	return predictedYabsolute



def computeTF(postText):
	terms = set(postText.split()) #use set to avoid double terms in results
	tfDict = {}
	for term in terms:
		tf = postText.count(term)
		tfDict[term] = tf

	return tfDict


def computeIDF(postText, sumBagOfWords, articleList):
	terms = set(postText.split()) #use set to avoid double terms in results
	idfDict = {}

	trainingDocuments = [open('../LDA-trainingFiles/'+file,'r').read() for file in os.listdir('../../LDA-Mallet GOOD/trainingFiles')]
	
	for term in terms:
		# number of documents containing term
		n = 0
		nDocumentsWithTerm = len([n + 1 for i in trainingDocuments if term in i])
		idf = math.log(len(trainingDocuments)/(1+nDocumentsWithTerm))
		idfDict[term] = idf

	return idfDict



def computeTFIDF(articleList, sumBagOfWords):
	top20wordsPerPost = {}
	for post in articleList:
		postID = post[0]
		postText = post[1]


		postTFdict = computeTF(postText)
		postIDFdict = computeIDF(postText, sumBagOfWords, articleList)

		tfidfScores = []
		for term in set(postText.split()): #use set to avoid double terms in results
			tfidf = postTFdict[term] * postIDFdict[term]
			tfidfScores.append((term,tfidf))

		tfidfScores.sort(key=lambda tup: tup[1], reverse=True)
		# print(tfidfScores)

		# save the top 15 most important words, if document in smaller, take all known words
		try:
			top20wordsPerPost[postID] = tfidfScores[:20]
		except:
			top20wordsPerPost[postID] = tfidfScores[:]

		# break

	return top20wordsPerPost


def predictTopic(title, summary, message):
	if os.path.exists('predictTopicFiles'):
		shutil.rmtree('predictTopicFiles')


	if not os.path.exists('predictTopicFiles'):
		os.makedirs('predictTopicFiles')

		with open('predictTopicFiles/predictTopicFile.txt', 'w') as predictTopicFile:
			predictTopicFile.write('{}\n{}\n{}'.format(title,summary,message))

		# get most important words of document with tf/idf
		postList = os.listdir('predictTopicFiles')
		articleList = [open('predictTopicFiles/predictTopicFile.txt','r').read()]  #list of texts
		# print(articleList[:2])


		bagOfWords = [collections.Counter(re.findall(r'\w+', post)) for post in articleList]
		sumBagOfWords = sum(bagOfWords, collections.Counter())

		top20wordsPerPost = computeTFIDF(articleList, sumBagOfWords)


		# remove tfidf scores
		for postID in top20wordsPerPost:
			wordList = []
			for pair in top20wordsPerPost[postID]:
				wordList.append(pair[0])
			top20wordsPerPost[postID] = wordList

		# use txt file with highest scoring words per topic to map those words to topic numbers
		officialTop20words = open('../LDA-MALLET/topicPerWordTop.txt', 'r').read().split('\n\n')
		officialTop20words = [i.split('\t') for i in officialTop20words]
		officialTop20words = [(i[0], i[1].split(') (')) for i in officialTop20words if i != ['']]

		officialTop20wordsList = []
		for i in officialTop20words:
			topicId = i[0]
			highestWords = []
			for n in i[1]:
				highestWords.append(n.split(',')[0].strip('(').strip("'"))
			officialTop20wordsList.append((topicId,highestWords))


		# assign most frequent topic number to post
		assignedTopicNumber = [] # [(postNr,topicNr), (postNr, topicNr), etc.]
		for postID in top20wordsPerPost:
			postWords = collections.Counter(top20wordsPerPost[postID])

			topicScores = []
			for item in officialTop20wordsList:
				topicNr = item[0]
				topicWords = collections.Counter(item[1])

				# postWord - topicWords --> what is left are words that do not match
				# the less words are left, the higher the similarity
				wordsLeft = list(postWords - topicWords)
				topicScores.append((topicNr,len(wordsLeft)))

			highestScoringTopicNr = min(topicScores, key=lambda t: t[1])[0]

		with open('../LDA-MALLET/topicsNew.txt','r') as topics:
			topics = topics.read().split('\n')
			topics = [topic.split('\t') for topic in topics]

			highestScoringTopic = [item[1] for item in topics if int(item[0]) == int(highestScoringTopicNr)][0]

	shutil.rmtree('predictTopicFiles')

	return highestScoringTopicNr, highestScoringTopic



def getTrainTest():
	trainData = pd.read_csv('allDataWithNormalizedReactions.csv', header=0,  sep=';')
	return trainData



def startText(summary, message, title):
	trainData = getTrainTest()

	topicNr, topic = predictTopic(title, summary, message)
	totalReactions = predictTotalReactions(trainData, title, summary, message)
	predictedReactionsNormalized, predictedReactionsAbsolute = predictReactionsVolume(trainData, title, summary, message, totalReactions)
	entropy, controversialLevel = predictEntropy(trainData, predictedReactionsAbsolute)

	return topicNr, topic, totalReactions, predictedReactionsNormalized, predictedReactionsAbsolute, entropy, controversialLevel








if __name__ == '__main__':
	main()