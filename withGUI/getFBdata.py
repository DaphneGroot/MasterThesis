from articleScraper import getArticleData
import os
import subprocess


def getFB(FBurl):

	postID = FBurl.split('/')[-1]
	source = FBurl.split('/')[-3]

	x = subprocess.Popen(['php', 'downloadFacebook.php', postID, source],stdout=subprocess.PIPE)
	output = x.stdout.readlines()

	message = output[0].strip().decode('utf-8')
	url = output[1].strip().decode('utf-8')

	return message, url



