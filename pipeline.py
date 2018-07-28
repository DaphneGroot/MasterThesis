import sys
# import csv
import pandas as pd
import numpy as np
import os
import shutil
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, f1_score
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
		probvect = [float(x)/sum(values) for x in values]

		# computer entropy over prob vector
		h = entropy(probvect)
	else:
		h = 0

	return h


def predictEntropy(trainData, testData):
	''' have to calculate entropy MSE'''
	# print(testData[-6:])
	testData['predictedEntropy'] = testData.iloc[:,-6:].apply(H, axis=1)

	mseEntropy = np.sqrt(mean_squared_error(testData['entropy'], testData['predictedEntropy']))
	print("MSE predicting entropy: ", mseEntropy)
	
	testData['controversialLevel'] = testData.iloc[:,-1].apply(lambda row: controversialLevel(row))

	return trainData, testData



def predictReactionsVolume(trainData,testData):

	'''per single reaction type'''
	reactions = ['normalizedLIKE','normalizedLOVE','normalizedHAHA','normalizedWOW','normalizedSAD','normalizedANGRY']
	
	for reaction in reactions:
		trainX = trainData["title"]+' '+trainData['summary']+' '+trainData['message']
		trainY = trainData[reaction]

		devX = testData["title"]+' '+testData['summary']+' '+testData['message']
		trueY = testData[reaction]



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
		predictedYnormalized = classifier.predict(devX)

		testData['predicted'+reaction] = predictedYnormalized


		MSE = mean_squared_error(trueY,predictedYnormalized)

		print("MSE predicting {} : {:<5.5f}".format(reaction, MSE))



	### Get absolute number of reactions

	absoluteReactionsPerRow = []
	for row in testData.iterrows():
		absoluteReactionsPerRow.append([row[1][-7:]])

	predictedReactionsAbsoluteAll = []
	for predictedReactionsNormalized in absoluteReactionsPerRow:
		predictedReactionsAbsolute = []

		predictedTotalReactionsAbsolute = predictedReactionsNormalized[0][0]
		predictedReactionsNormalized = predictedReactionsNormalized[0][1:]

		negatives = np.array(predictedReactionsNormalized) < 0
		if True in negatives:
			for idx,i in enumerate(negatives):
				if i == True:
					numberToDivide = predictedReactionsNormalized[idx]
					subtractPerNumber = numberToDivide/list(negatives).count(False)

					for idx2, n in enumerate(negatives):
						if n == False:
							predictedReactionsNormalized[idx2] = predictedReactionsNormalized[idx2]-subtractPerNumber
						
					predictedReactionsNormalized[idx] = 0.0

		# convert to absolute numbers
		for i in predictedReactionsNormalized:
			x = i * predictedTotalReactionsAbsolute
			predictedYabsolute = int(round(x))
			predictedReactionsAbsolute.append(predictedYabsolute)

		predictedReactionsAbsoluteAll.append(predictedReactionsAbsolute)
		
		# break

	likeAbsolute = []
	loveAbsolute = []
	hahaAbsolute = []
	wowAbsolute = []
	sadAbsolute = []
	angryAbsolute = []
	for i in predictedReactionsAbsoluteAll:
		likeAbsolute.append(i[0])
		loveAbsolute.append(i[1])
		hahaAbsolute.append(i[2])
		wowAbsolute.append(i[3])
		sadAbsolute.append(i[4])
		angryAbsolute.append(i[5])

	testData['predictedLIKEabsolute'] = likeAbsolute
	testData['predictedLOVEabsolute'] = loveAbsolute
	testData['predictedHAHAabsolute'] = hahaAbsolute
	testData['predictedWOWabsolute'] = wowAbsolute
	testData['predictedSADabsolute'] = sadAbsolute
	testData['predictedANGRYabsolute'] = angryAbsolute


	return trainData, testData


def predictTotalReactions(trainData,testData):
	'''using normalized scores:'''
	trainX = trainData["title"]+' '+trainData['summary']+' '+trainData['message']
	trainY = trainData['normalizedTotalReactions']

	devX = testData["title"]+' '+testData['summary']+' '+testData['message']
	trueY = testData['normalizedTotalReactions']

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

	predictedY = classifier.predict(devX)

	MSE = mean_squared_error(trueY,predictedY)
	print("MSE predicting total reactions normalized: ", MSE)

	# score = classifier.score(devX,trueY)
	# print("Score predicting total reactions: ", score)

	predictedYabsolute = [int(round(10**i)) for i in predictedY]

	testData['predictedNormalizedTotalReactions'] = predictedY
	testData['predictedAbsoluteTotalReactions'] = predictedYabsolute

	return testData


def predictTopicMallet(testData):
	predictedTopicNrList = []
	predictedTopicList = []

	if os.path.exists('predictTopicFiles'):
		shutil.rmtree('predictTopicFiles')
		
	if not os.path.exists('predictTopicFiles'):
		os.makedirs('predictTopicFiles')

		for index, row in testData.iterrows():
			title = row[13]
			summary = row[15]
			message = row[0]
			idNumber = testData[testData['title']==title].index[0]

			with open('predictTopicFiles/predictTopicFile_'+str(idNumber)+'_.txt', 'w') as predictTopicFile:
				predictTopicFile.write('{}\n{}\n{}'.format(title,summary,message))

		os.chdir('LDA-MALLET/mallet-2.0.8') # change to Mallet directory
		os.system("bin/mallet import-dir --input ../../Pipeline/predictTopicFiles --output files-to-predict.mallet --keep-sequence --stoplist-file stoplists/nl.txt --use-pipe-from topic-input.mallet")
		os.system('bin/mallet infer-topics --input files-to-predict.mallet --inferencer inferencer.mallet --output-doc-topics new-topic-composition.txt')

		with open('new-topic-composition.txt','r') as newTopicCompositionFile:
			newTopicCompositionFile = newTopicCompositionFile.read().split('\n')[1:-1] #skip first line with headers
			for entry in newTopicCompositionFile:
				rowID = entry.split('_')[1]

				topicScores = entry.split()[7:]
				topicScores = [float(i) for i in topicScores]
				highestScoringTopicNr = topicScores.index(max(topicScores))
				predictedTopicNrList.append((rowID,highestScoringTopicNr))

				with open('../topicsNew.txt','r') as topics:
					topics = topics.read().split('\n')
					topics = [topic.split('\t') for topic in topics]

					highestScoringTopic = [item[1] for item in topics if int(item[0]) == highestScoringTopicNr][0]
					predictedTopicList.append((rowID,highestScoringTopic))

		os.chdir('../../') # change to original directory
	shutil.rmtree('predictTopicFiles')

	return predictedTopicNrList
	
def computeTF(postText):
	terms = set(postText.split()) #use set to avoid double terms in results
	tfDict = {}
	for term in terms:
		tf = postText.count(term)
		tfDict[term] = tf

	return tfDict


def computeIDF(postText, sumBagOfWords, articleList, trainingDocuments):
	terms = set(postText.split()) #use set to avoid double terms in results
	idfDict = {}

	

	for term in terms:
		# number of documents containing term
		n = 0
		nDocumentsWithTerm = len([n + 1 for i in trainingDocuments if term in i])
		idf = math.log(len(trainingDocuments)/(1+nDocumentsWithTerm))
		idfDict[term] = idf

	return idfDict



def computeTFIDF(articleList, sumBagOfWords):
	top10wordsPerPost = {}
	
	trainingDocuments = [open('LDA-trainingFiles/'+file,'r').read() for file in os.listdir('../LDA-Mallet GOOD/trainingFiles')]

	for post in articleList:
		postID = post[0]
		postText = post[1]


		postTFdict = computeTF(postText)
		postIDFdict = computeIDF(postText, sumBagOfWords, articleList, trainingDocuments)

		tfidfScores = []
		for term in set(postText.split()): #use set to avoid double terms in results
			tfidf = postTFdict[term] * postIDFdict[term]
			tfidfScores.append((term,tfidf))

		tfidfScores.sort(key=lambda tup: tup[1], reverse=True)
		# print(tfidfScores)

		# save the top 15 most important words, if document in smaller, take all known words
		try:
			top10wordsPerPost[postID] = tfidfScores[:20]
		except:
			top10wordsPerPost[postID] = tfidfScores[:]

		# break

	return top10wordsPerPost



def predictTopicTfIdf(testData):
	predictedTopicNrList = []
	predictedTopicList = []

	if os.path.exists('predictTopicFiles'):
		shutil.rmtree('predictTopicFiles')
		
	if not os.path.exists('predictTopicFiles'):
		os.makedirs('predictTopicFiles')

		for index, row in testData.iterrows():
			title = row[13]
			summary = row[15]
			message = row[0]
			idNumber = testData[testData['title']==title].index[0]

			with open('predictTopicFiles/predictTopicFile_'+str(idNumber)+'_.txt', 'w') as predictTopicFile:
				predictTopicFile.write('{}\n{}\n{}'.format(title,summary,message))


		# get most important words of document with tf/idf
		postList = os.listdir('predictTopicFiles')
		articleList = [(i.split('_')[1],open('predictTopicFiles/'+i,'r').read()) for i in postList] #list of texts
		# print(articleList[:2])

		bagOfWords = [collections.Counter(re.findall(r'\w+', post[1])) for post in articleList]
		sumBagOfWords = sum(bagOfWords, collections.Counter())

		top10wordsPerPost = computeTFIDF(articleList, sumBagOfWords)

		# remove tfidf scores
		for postID in top10wordsPerPost:
			wordList = []
			for pair in top10wordsPerPost[postID]:
				wordList.append(pair[0])
			top10wordsPerPost[postID] = wordList

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


		# assign most frequent topic number to post
		assignedTopicNumber = [] # [(postNr,topicNr), (postNr, topicNr), etc.]
		for postID in top10wordsPerPost:
			postWords = collections.Counter(top10wordsPerPost[postID])

			topicScores = []
			for item in officialTop20wordsList:
				topicNr = item[0]
				topicWords = collections.Counter(item[1])

				# postWord - topicWords --> what is left are words that do not match
				# the less words are left, the higher the similarity
				wordsLeft = list(postWords - topicWords)
				topicScores.append((topicNr,len(wordsLeft)))

			highestScoringTopicNr = min(topicScores, key=lambda t: t[1])[0]
			assignedTopicNumber.append((postID, int(highestScoringTopicNr)))

	shutil.rmtree('predictTopicFiles')

	return assignedTopicNumber


def predictTopicStart(trainData,testData):
	''' MALLET '''
	predictedTopicNrList = predictTopicMallet(testData)

	# matched on title --> assumed the title is always unique
	testData['predictedTopic'] = ""
	for index, row in testData.iterrows():
		rowID = int(testData[testData['title']==row[13]].index[0])
		predictedTopicNr = [predictedTopic[1] for predictedTopic in predictedTopicNrList if int(predictedTopic[0]) == rowID]
		testData.at[rowID,'predictedTopic'] = predictedTopicNr[0]

	mseTopic = np.sqrt(mean_squared_error(testData['highestScoringTopic'], testData['predictedTopic']))
	print("\nMSE predicting topic MALLET:", mseTopic)

	mseTopic = np.sqrt(f1_score(list(testData['highestScoringTopic']), list(testData['predictedTopic']), average='micro'))
	print("F1 predicting topic MALLET:", mseTopic)

	mseTopic = np.sqrt(accuracy_score(list(testData['highestScoringTopic']), list(testData['predictedTopic'])))
	print("Accuracy predicting topic MALLET:", mseTopic)
	print()


	''' TF-IDF '''
	predictedTopicNrList = predictTopicTfIdf(testData)

		# matched on title --> assumed the title is always unique
	testData['predictedTopic'] = ""
	for index, row in testData.iterrows():
		rowID = int(testData[testData['title']==row[13]].index[0])
		predictedTopicNr = [predictedTopic[1] for predictedTopic in predictedTopicNrList if int(predictedTopic[0]) == rowID]
		testData.at[rowID,'predictedTopic'] = predictedTopicNr[0]

	mseTopic = np.sqrt(mean_squared_error(testData['highestScoringTopic'], testData['predictedTopic']))
	print("MSE predicting topic TF-IDF: ", mseTopic)

	mseTopic = np.sqrt(f1_score(list(testData['highestScoringTopic']), list(testData['predictedTopic']), average='micro'))
	print("F1 predicting topic TF-IDF: ", mseTopic)

	mseTopic = np.sqrt(accuracy_score(list(testData['highestScoringTopic']), list(testData['predictedTopic'])))
	print("Accuracy predicting topic TF-IDF:", mseTopic)
	print()

	return testData
	


def getTrainTest():
	allData = pd.read_csv('allDataWithNormalizedReactions.csv', header=0,  sep=';')
	allData = allData.sample(frac=1, random_state=1) # use to suffle dataset

	lenTrain = round(len(allData) * 0.8)
	lenTest = round(len(allData) * 0.2)

	trainData = allData[:lenTrain]
	testData = allData[lenTrain:]

	
	return trainData, testData, allData


def main():
	pd.options.mode.chained_assignment = None
	trainData, testData, allData = getTrainTest()

	testData = predictTopicStart(trainData, testData)
	testData = predictTotalReactions(trainData, testData)
	trainData, testData = predictReactionsVolume(trainData, testData)
	trainData, testData = predictEntropy(trainData, testData)



if __name__ == '__main__':
	main()