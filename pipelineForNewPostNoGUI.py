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
    if sum(values) != 0: #to prevent 'division by 0'
        probvect = [int(x)/sum(values) for x in values]

        # computer entropy over prob vector
        h = entropy(probvect)
    else:
        h = 0

    return h


def predictEntropy(trainData, predictedReactionsVolume):
	predictedEntropy = H(predictedReactionsVolume)
	print('Predcited entropy score: ', predictedEntropy)


	controversial = controversialLevel(predictedEntropy)
	print('Predicted controversial level: ', controversial)


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


	for idx, reaction in enumerate(['LIKE','LOVE','HAHA','WOW','SAD','ANGRY']):
		print("Roughly (!) predicted", reaction, ": ",predictedReactionsAbsolute[idx])

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
	
	print("\nRoughly (!) predicted total number of reactions: ", predictedYabsolute)

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

	# open traing documents (same training documents as from MALLET)	
	# trainingDocuments = open('../../LDA-Mallet GOOD/trainingFiles')

	trainingDocuments = [open('LDA-trainingFiles/'+file,'r').read() for file in os.listdir('LDA-trainingFiles')]
	# print(trainingDocuments)

	for term in terms:
		# number of documents containing term
		n = 0
		nDocumentsWithTerm = len([n + 1 for i in trainingDocuments if term in i]) #based on trainingfiles
		idf = math.log(len(trainingDocuments)/(1+nDocumentsWithTerm))
		idfDict[term] = idf

	return idfDict



def computeTFIDF(articleList, sumBagOfWords):
	top20wordsPerPost = {}
	for post in articleList:
		postID = 1
		postText = post


		postTFdict = computeTF(postText)
		postIDFdict = computeIDF(postText, sumBagOfWords, articleList)

		tfidfScores = []
		for term in set(postText.split()): #use set to avoid double terms in results
			tfidf = postTFdict[term] * postIDFdict[term]
			tfidfScores.append((term,tfidf))

		tfidfScores.sort(key=lambda tup: tup[1], reverse=True)

		# save the top 20 most important words, if document in smaller, take all known words
		try:
			top20wordsPerPost[postID] = tfidfScores[:20]
		except:
			top20wordsPerPost[postID] = tfidfScores[:]


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
		officialTop20words = open('LDA-MALLET/topicPerWordTop.txt', 'r').read().split('\n\n')
		officialTop20words = [i.split('\t') for i in officialTop20words]
		officialTop20words = [(i[0], i[1].split(') (')) for i in officialTop20words if i != ['']]

		officialTop20wordsList = []
		for i in officialTop20words:
			topicId = i[0]
			highestWords = []
			for n in i[1]:
				highestWords.append(n.split(',')[0].strip('(').strip("'"))
			officialTop20wordsList.append((topicId,highestWords))


		# print(officialTop20wordsList)

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

			print("\nHighest scoring topic number: ", highestScoringTopicNr)
			# print("Highest scoring topic: ", highestScoringTopic)

	shutil.rmtree('predictTopicFiles')



def getTrainTest():
	trainData = pd.read_csv('allDataWithNormalizedReactions.csv', header=0,  sep=';')
	return trainData


def getVariables():
	parts = ['title','summary','message']
	print("\033[93m"+'\n\tUse `|` to go to the next variable'+"\033[0m")


	title = ''
	summary = ''
	message = ''

	for part in parts:
		print('\nPlease type the \033[91m',part,'\033[0m:')
		for item in sys.stdin:
			
			item = item.strip()

			if part == 'title':
				if item != '|':
					title = title + ' ' +item

				if title != '':
					if item == '|' or '|' in item:
						break
				else:
					print('You haven\'t sumbmitted a title!')
				

			elif part == 'summary':
				if item != '|':
					summary = summary + ' ' + item

				if summary != '':
					if item == '|' or '|' in item:
						break
				else:
					print('You haven\'t sumbmitted a summary!')

			else:
				if item != '|':
					message = message + ' ' + item

				if message != '':
					if item == '|' or '|' in item:
						break
				else:
					print('You haven\'t sumbmitted a message!')


	# remove added |
	title = title.strip('|')
	summary = summary.strip('|')
	message = message.strip('|')

	# print('\n\ntitle: ', title)
	# print('summary: ', summary)
	# print('message: ', message)

	return title, summary, message


def main():
	# use this when you want to type in your own variables, and do not want to type something
	# title, summary, message = getVariables()

	# use this when you just want to try, and do not want to type something
	trainData = getTrainTest()
	title = "Ik wil niet dat mijn man gemasseerd wordt door een vrouw - wat zou u doen?"
	summary = "artikelLezers vragen andere lezers om raad bij problemen en dilemma's."
	message = "'Mijn man heeft sinds kort een nieuwe masseuse, maar heeft me daar niet over ingelicht. Dat vind ik stiekem.' Wat zou u doen? Lees hier wat andere lezers deze vrouw adviseren."

	print('\n---------------------------\n')

	predictTopic(title, summary, message)
	totalReactions = predictTotalReactions(trainData, title, summary, message)
	predictedReactionsPercentage, predictedReactionsAbsolute = predictReactionsVolume(trainData, title, summary, message, totalReactions)
	predictEntropy(trainData, predictedReactionsAbsolute)










if __name__ == '__main__':
	main()