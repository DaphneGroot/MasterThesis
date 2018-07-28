from PyQt4 import QtGui, QtCore
import sys  
from pipelineForNewPost import startText
from articleScraper import getArticleData
from getFBdata import getFB

class Window(QtGui.QWidget):
	"""Generates main window and navigates between screens"""

	def __init__(self):
		super(Window, self).__init__()

		self.startscreen = StartScreen(self)
		self.resize(300, 400)
		self.center()
		# self.setGeometry(550,150,400,500) # links/rechts, boven/beneden, breedte, hoogte
		self.setWindowTitle("Pick a method")
		self.setWindowIcon(QtGui.QIcon('Images/icon.jpg')) 


		self.setStyleSheet("""
			QWidget {
				background-color: qlineargradient( x1:0 y1:1, x2:0 y2:0, stop:0 #D4C3C3, stop:1 white);
			}

			QPushButton {
				background-color: #E9E7DE;
				color: black;
				font-family: Verdana;
				font-size: 10px;
				border: 1px solid black;
				
			}
			QLabel {
				background:transparent;
				font-family: Verdana;
				font-size: 10px;
			}

			QPlainTextEdit {
				background-color: white;
				font-family: Verdana;
				font-size: 10px;
			}
			""")

		self.show()

	def center(self):
		qr = self.frameGeometry()
		cp = QtGui.QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def toStartScreen(self):
		self.startscreen = StartScreen(self)
		self.startscreen.show()

	def toFBscreen(self):
		self.fbscreen = FBscreen(self)
		self.fbscreen.show()

	def toURLscreen(self):
		self.urlscreen = URLscreen(self)
		self.urlscreen.show()

	def toTEXTscreen(self):
		self.textscreen = TEXTscreen(self)
		self.textscreen.show()

	def toResultscreen(self, topicNr, topic, totalReactions, predictedReactionsNormalized, predictedReactionsAbsolute, entropy, controversialLevel):
		self.resultscreen = resultScreen(self, topicNr, topic, totalReactions, predictedReactionsNormalized, predictedReactionsAbsolute, entropy, controversialLevel)
		self.resultscreen.show()

##################################################

class StartScreen(QtGui.QWidget):
	"""Generates start screen"""

	def __init__(self, parent):
		super(StartScreen, self).__init__(parent)
		self.parent = parent
		self.setGeometry(0,0,self.parent.frameGeometry().width(),self.parent.frameGeometry().height())
		self.parent.setWindowTitle("Pick a method")

		self.FBButton = QtGui.QPushButton("Use Facebook link", self)
		self.articleButton = QtGui.QPushButton("Use article link", self)
		self.textButton = QtGui.QPushButton("Use manual added text", self)

		self.FBButton.resize(140,90)
		self.articleButton.resize(140,90)
		self.textButton.resize(140,90)

		self.FBButton.move(85,40)
		self.articleButton.move(85,150)
		self.textButton.move(85,260)

		self.textButton.clicked.connect(lambda: self.pickMethod("text"))
		self.articleButton.clicked.connect(lambda: self.pickMethod("article"))
		self.FBButton.clicked.connect(lambda: self.pickMethod("fb"))

	def pickMethod(self,method):
			if method == "text":
				self.parent.toTEXTscreen()
			elif method == 'article':
				self.parent.toURLscreen()
			else:
				self.parent.toFBscreen()

			self.close()



class FBscreen(QtGui.QWidget):
	"""Generates facebook-input screen"""

	def __init__(self, parent):
		super(FBscreen, self).__init__(parent)
		self.parent = parent

		self.setGeometry(0,0,self.parent.frameGeometry().width(),self.parent.frameGeometry().height())
		self.parent.setWindowTitle("URL Facebook")

		self.boldFont = QtGui.QFont()
		self.boldFont.setBold(True)

		self.explanationLabel = QtGui.QLabel("Please give the URL of the Facebook post:",self)
		self.explanationLabel.setFont(self.boldFont)
		self.explanationLabel.resize(200, 50)
		self.explanationLabel.setWordWrap(True)
		self.explanationLabel.move(50,15)

		self.noteLabel = QtGui.QLabel("Note: please note that for now, this function only works with links from Facebookpages of 'Parool', 'Nu.nl', 'NRC', 'Telegraaf', 'NOS' and 'Volkskrant'.\nOther sources will follow.",self)
		self.noteLabel.resize(200, 150)
		self.noteLabel.setWordWrap(True)
		self.noteLabel.move(50,30)

		self.FBurl = QtGui.QPlainTextEdit(self)
		self.FBurl.resize(200,80)
		self.FBurl.move(50,180)

		self.sendButton = QtGui.QPushButton("Show predictions", self)
		self.sendButton.resize(200, 60)
		self.sendButton.move(50,280)
		self.sendButton.clicked.connect(lambda: self.changeButton(self.sendButton,self.FBurl))

		self.toStartButton = QtGui.QPushButton("Back to start", self)
		self.toStartButton.resize(70, 15)
		self.toStartButton.move(0,0)
		self.toStartButton.clicked.connect(lambda: self.startButton(self.toStartButton))


	def startButton(self,toStartButton):
		self.parent.toStartScreen()
		self.close()
		
	def changeButton(self,sendButton,FBurl):
		self.sendButton.setText("The data is being processed.\nPlease wait...")	
		self.sendButton.repaint()	
		self.getFBDataFunction(self.FBurl)

	def getFBDataFunction(self,FBurl):
		try:
			self.FBURLtext = self.FBurl.toPlainText()
			self.source = self.FBURLtext.split('/')[-3]
		except:
			self.sendButton.setText("Show predictions")	
			self.sendButton.repaint()
			self.warning = MyPopupWarning()
			self.warning.setGeometry(670, 300, 200, 100)
			self.warning.show()

		if self.FBURLtext == "":
			self.sendButton.setText("Show predictions")	
			self.sendButton.repaint()
			self.warning = MyPopupWarningFB()
			self.warning.setGeometry(670, 300, 200, 100)
			self.warning.show()

		elif 'posts' not in self.FBURLtext:
			self.sendButton.setText("Show predictions")	
			self.sendButton.repaint()
			self.warning = MyPopupWarningFB()
			self.warning.setGeometry(670, 300, 200, 100)
			self.warning.show()

		elif self.source not in ['paroolnl','nu','nrc','telegraaf','nos','volkskrant']:
			self.sendButton.setText("Show predictions")	
			self.sendButton.repaint()
			self.warning = MyPopupWarningSoure()
			self.warning.setGeometry(670, 300, 200, 100)
			self.warning.show()

		elif self.FBURLtext != "":
			self.message, self.url = getFB(self.FBURLtext)
			self.title, self.summary = getArticleData(self.url)
			self.FBurl.clear()

			self.topicNr, self.topic, self.totalReactions, self.predictedReactionsNormalized, self.predictedReactionsAbsolute, self.entropy, self.controversialLevel = startText(self.summary, self.message, self.title)
			self.parent.toResultscreen(self.topicNr, self.topic, self.totalReactions, self.predictedReactionsNormalized, self.predictedReactionsAbsolute, self.entropy, self.controversialLevel)
			self.close()
			
		else:
			self.sendButton.setText("Show predictions")	
			self.sendButton.repaint()
			self.warning = MyPopupWarning()
			self.warning.setGeometry(670, 300, 200, 100)
			self.warning.show()



class URLscreen(QtGui.QWidget):
	"""Generates article-input screen"""

	def __init__(self, parent):
		super(URLscreen, self).__init__(parent)
		self.parent = parent

		self.setGeometry(0,0,self.parent.frameGeometry().width(),self.parent.frameGeometry().height())
		self.parent.setWindowTitle("URL article")

		self.boldFont = QtGui.QFont()
		self.boldFont.setBold(True)

		self.explanationLabel = QtGui.QLabel("Please give the URL of the article:",self)
		self.explanationLabel.setFont(self.boldFont)
		self.explanationLabel.resize(200, 50)
		self.explanationLabel.setWordWrap(True)
		self.explanationLabel.move(50,15)

		self.noteLabel = QtGui.QLabel("Note 1: please remember that the message is not provided, so the results may be inaccurate!\n\nNote 2: please note that for now, this function only works with links from 'www.parool.nl', 'www.nu.nl', 'www.nrc.nl', 'www.telegraaf.nl', 'www.nos.nl' and 'www.volkskrant.nl'.\nOther sources will follow.",self)
		self.noteLabel.resize(200, 150)
		self.noteLabel.setWordWrap(True)
		self.noteLabel.move(50,30)

		self.articleURL = QtGui.QPlainTextEdit(self)
		self.articleURL.resize(200,80)
		self.articleURL.move(50,180)

		self.sendButton = QtGui.QPushButton("Show predictions", self)
		self.sendButton.resize(200, 60)
		self.sendButton.move(50,280)
		self.sendButton.clicked.connect(lambda: self.changeButton(self.sendButton,self.articleURL))
		
		self.toStartButton = QtGui.QPushButton("Back to start", self)
		self.toStartButton.resize(70, 15)
		self.toStartButton.move(0,0)
		self.toStartButton.clicked.connect(lambda: self.startButton(self.toStartButton))


	def startButton(self,toStartButton):
		self.parent.toStartScreen()
		self.close()


	def changeButton(self,sendButton,articleURL):
		self.sendButton.setText("The data is being processed.\nPlease wait...")		
		self.sendButton.repaint()
		self.getArticleDataFunction(self.articleURL)

	def getArticleDataFunction(self,articleURL):
		try:
			self.articleURLtext = self.articleURL.toPlainText()
			self.source = self.articleURLtext.split('.')[1]
		except:
			self.sendButton.setText("Show predictions")	
			self.sendButton.repaint()
			self.warning = MyPopupWarning()
			self.warning.setGeometry(670, 300, 200, 100)
			self.warning.show()


		if self.articleURLtext == "":
			self.sendButton.setText("Show predictions")	
			self.sendButton.repaint()
			self.warning = MyPopupWarningFB()
			self.warning.setGeometry(670, 300, 200, 100)
			self.warning.show()

		elif self.source not in ['parool','nu','nrc','telegraaf','nos','volkskrant']:
			self.sendButton.setText("Show predictions")	
			self.sendButton.repaint()
			self.warning = MyPopupWarningSoure()
			self.warning.setGeometry(670, 300, 200, 100)
			self.warning.show()

		elif self.articleURLtext != "":
			self.title, self.summary = getArticleData(self.articleURLtext)
			self.message = "No Message"

			self.articleURL.clear()

			self.topicNr, self.topic, self.totalReactions, self.predictedReactionsNormalized, self.predictedReactionsAbsolute, self.entropy, self.controversialLevel = startText(self.summary, self.message, self.title)
			self.parent.toResultscreen(self.topicNr, self.topic, self.totalReactions, self.predictedReactionsNormalized, self.predictedReactionsAbsolute, self.entropy, self.controversialLevel)
			self.close()

		else:
			self.sendButton.setText("Show predictions")	
			self.sendButton.repaint()
			self.warning = MyPopupWarning()
			self.warning.setGeometry(670, 300, 200, 100)
			self.warning.show()



class TEXTscreen(QtGui.QWidget):
	"""Generates text-input screen"""
	def __init__(self, parent):
		super(TEXTscreen, self).__init__(parent)
		self.parent = parent

		self.setGeometry(0,0,self.parent.frameGeometry().width(),self.parent.frameGeometry().height())
		self.parent.setWindowTitle("Article text")

		self.toStartButton = QtGui.QPushButton("Back to start", self)
		self.toStartButton.resize(70, 15)
		self.toStartButton.move(0,0)
		self.toStartButton.clicked.connect(lambda: self.startButton(self.toStartButton))

		self.boldFont = QtGui.QFont()
		self.boldFont.setBold(True)

		self.explanationLabel = QtGui.QLabel("Fill in the field below with the correct texts.\nYou cannot leave a field empty.",self)
		self.explanationLabel.setFont(self.boldFont)
		self.explanationLabel.resize(200, 50)
		self.explanationLabel.setWordWrap(True)
		self.explanationLabel.move(55,15)

		self.messageLabel = QtGui.QLabel("Type the message of the post:",self)
		self.messageLabel.adjustSize()
		self.messageLabel.move(45,60)
		self.messageField = QtGui.QPlainTextEdit(self)
		self.messageField.resize(220, 60)
		self.messageField.move(45,75)

		self.summaryLabel = QtGui.QLabel("Type the summary of the post:",self)
		self.summaryLabel.adjustSize()
		self.summaryLabel.move(45,145)
		self.summaryField = QtGui.QPlainTextEdit(self)
		self.summaryField.resize(220, 60)
		self.summaryField.move(45,160)

		self.titleLabel = QtGui.QLabel("Type the title of the post:",self)
		self.titleLabel.adjustSize()
		self.titleLabel.move(45,230)
		self.titleField = QtGui.QPlainTextEdit(self)
		self.titleField.resize(220, 60)
		self.titleField.move(45,245)

		self.sendButton = QtGui.QPushButton("Show predictions", self)
		self.sendButton.resize(220, 50)
		self.sendButton.move(45,315)
		self.sendButton.clicked.connect(lambda: self.changeButton(self.sendButton,self.messageField,self.summaryField,self.titleField))


	def startButton(self,toStartButton):
		self.parent.toStartScreen()
		self.close()

	def changeButton(self,sendButton,messageField,summaryField,titleField):
		self.sendButton.setText("The data is being processed, please wait...")
		self.sendButton.repaint()	
		self.printVariables(self.messageField,self.summaryField,self.titleField)


	def printVariables(self, messageField, summaryField, titleField):
		self.messageText = self.messageField.toPlainText()
		self.summaryText = self.summaryField.toPlainText()
		self.titleText = self.titleField.toPlainText()


		if self.messageText != "" and self.summaryText != "" and self.titleText != "":
			self.messageField.clear()
			self.summaryField.clear()
			self.titleField.clear()

			self.topicNr, self.topic, self.totalReactions, self.predictedReactionsNormalized, self.predictedReactionsAbsolute, self.entropy, self.controversialLevel = startText(self.summaryText, self.messageText, self.titleText)
			self.parent.toResultscreen(self.topicNr, self.topic, self.totalReactions, self.predictedReactionsNormalized, self.predictedReactionsAbsolute, self.entropy, self.controversialLevel)
			self.close()

		else:
			self.sendButton.setText("Show predictions")	
			self.sendButton.repaint()
			self.warning = MyPopupWarning()
			self.warning.setGeometry(670, 300, 200, 100)
			self.warning.show()



class MyPopupWarning(QtGui.QWidget):
	def __init__(self):
		QtGui.QWidget.__init__(self)
		self.warningLabel = QtGui.QLabel("\n\n   Fill in all the fields correctly!",self)

		self.setWindowTitle("Warning!")
		self.setWindowIcon(QtGui.QIcon('Images/icon.jpg')) 


		self.setStyleSheet("""
			QWidget {
				background-color: qlineargradient( x1:0 y1:1, x2:0 y2:0, stop:0 #9F9292, stop:1 white);
			}

			QLabel {
				background:transparent;
				font-family: Verdana;
				font-size: 11px;
			}
			""")


class MyPopupWarningFB(QtGui.QWidget):
	def __init__(self):
		QtGui.QWidget.__init__(self)
		self.warningLabel = QtGui.QLabel("\n\n   Please enter a Facebook link that\n   contains a link to an article!",self)
		# self.warningLabel.adjustSize()
		# self.warningLabel.setWordWrap(True)

		self.setWindowTitle("Warning!")
		self.setWindowIcon(QtGui.QIcon('Images/icon.jpg')) 


		self.setStyleSheet("""
			QWidget {
				background-color: qlineargradient( x1:0 y1:1, x2:0 y2:0, stop:0 #9F9292, stop:1 white);
			}

			QLabel {
				background:transparent;
				font-family: Verdana;
				font-size: 11px;
			}
			""")


class MyPopupWarningSoure(QtGui.QWidget):
	def __init__(self):
		QtGui.QWidget.__init__(self)
		self.warningLabel = QtGui.QLabel("\n   Please enter a link from the\n   following sources:\n   Parool, NU.nl, NRC, Telegraaf,\n   NOS or Volkskrant",self)
		# self.warningLabel.adjustSize()
		# self.warningLabel.setWordWrap(True)

		self.setWindowTitle("Warning!")
		self.setWindowIcon(QtGui.QIcon('Images/icon.jpg')) 


		self.setStyleSheet("""
			QWidget {
				background-color: qlineargradient( x1:0 y1:1, x2:0 y2:0, stop:0 #9F9292, stop:1 white);
			}

			QLabel {
				background:transparent;
				font-family: Verdana;
				font-size: 11px;
			}
			""")


class resultScreen(QtGui.QWidget):
	"""Generates results screen"""

	def __init__(self, parent, topicNr, topic, totalReactions, predictedReactionsNormalized, predictedReactionsAbsolute, entropy, controversialLevel):
		super(resultScreen, self).__init__(parent)
		self.parent = parent

		self.setGeometry(0,0,self.parent.frameGeometry().width(),self.parent.frameGeometry().height())
		self.parent.setWindowTitle("Results")

		self.topicNr = topicNr
		self.topic = topic
		self.totalReactions = totalReactions
		self.predictedReactionsAbsoluteLIKE = str(predictedReactionsAbsolute[0])
		self.predictedReactionsAbsoluteLOVE = str(predictedReactionsAbsolute[1])
		self.predictedReactionsAbsoluteHAHA = str(predictedReactionsAbsolute[2])
		self.predictedReactionsAbsoluteWOW = str(predictedReactionsAbsolute[3])
		self.predictedReactionsAbsoluteSAD = str(predictedReactionsAbsolute[4])
		self.predictedReactionsAbsoluteANGRY = str(predictedReactionsAbsolute[5])

		self.predictedReactionsNormalizedLIKE = str(round(predictedReactionsNormalized[0][0],3))
		self.predictedReactionsNormalizedLOVE = str(round(predictedReactionsNormalized[1][0],3))
		self.predictedReactionsNormalizedHAHA = str(round(predictedReactionsNormalized[2][0],3))
		self.predictedReactionsNormalizedWOW = str(round(predictedReactionsNormalized[3][0],3))
		self.predictedReactionsNormalizedSAD = str(round(predictedReactionsNormalized[4][0],3))
		self.predictedReactionsNormalizedANGRY = str(round(predictedReactionsNormalized[5][0],3))
		# self.predictedReactionsPercentage = predictedReactionsPercentage
		self.entropy = float(entropy)
		self.controversialLevel = controversialLevel

		self.setGeometry(0,0,self.parent.frameGeometry().width(),self.parent.frameGeometry().height())

		self.boldFont = QtGui.QFont()
		self.boldFont.setBold(True)

		self.explanationLabel = QtGui.QLabel("Roughly predicted scores:",self)
		self.explanationLabel.setFont(self.boldFont)
		self.explanationLabel.resize(200, 50)
		self.explanationLabel.setWordWrap(True)
		self.explanationLabel.move(55,0)

		self.topicLabel = QtGui.QLabel("Topic: "+ self.topic + " (topic number = "+ str(self.topicNr) +")",self)
		self.topicLabel.resize(200, 50)
		self.topicLabel.setWordWrap(True)
		self.topicLabel.move(55,15)

		self.totalReactionsLabel = QtGui.QLabel("Total number of reactions: "+ str(self.totalReactions),self)
		self.totalReactionsLabel.resize(200, 50)
		self.totalReactionsLabel.setWordWrap(True)
		self.totalReactionsLabel.move(55,40)

		self.LIKEimage = QtGui.QPixmap('Images/like.png')
		self.LIKEimage = self.LIKEimage.scaledToWidth(20)
		self.LIKElabelimage = QtGui.QLabel(self)
		self.LIKElabelimage.resize(40, 40)
		self.LIKElabelimage.move(80,65)
		self.LIKElabelimage.setPixmap(self.LIKEimage)
		# self.LIKElabel = QtGui.QLabel(str(self.predictedReactionsAbsolute[0])+" ("+str(self.predictedReactionsPercentage[0])+"%)",self)
		self.LIKElabel = QtGui.QLabel(str(self.predictedReactionsAbsoluteLIKE)+" (norm.: "+str(self.predictedReactionsNormalizedLIKE)+")",self)
		self.LIKElabel.resize(100, 50)
		self.LIKElabel.move(105,60)

		self.LOVEimage = QtGui.QPixmap('Images/love.png')
		self.LOVEimage = self.LOVEimage.scaledToWidth(20)
		self.LOVElabelimage = QtGui.QLabel(self)
		self.LOVElabelimage.resize(40, 40)
		self.LOVElabelimage.move(80,90)
		self.LOVElabelimage.setPixmap(self.LOVEimage)
		# self.LOVElabel = QtGui.QLabel(str(self.predictedReactionsAbsolute[1])+" ("+str(self.predictedReactionsPercentage[1])+"%)",self)
		self.LOVElabel = QtGui.QLabel(str(self.predictedReactionsAbsoluteLOVE)+" (norm.: "+str(self.predictedReactionsNormalizedLOVE)+")",self)
		self.LOVElabel.resize(100, 50)
		self.LOVElabel.move(105,85)

		self.HAHAimage = QtGui.QPixmap('Images/haha.png')
		self.HAHAimage = self.HAHAimage.scaledToWidth(20)
		self.HAHAlabelimage = QtGui.QLabel(self)
		self.HAHAlabelimage.resize(40, 40)
		self.HAHAlabelimage.move(80,115)
		self.HAHAlabelimage.setPixmap(self.HAHAimage)
		# self.HAHAlabel = QtGui.QLabel(str(self.predictedReactionsAbsolute[2])+" ("+str(self.predictedReactionsPercentage[2])+"%)",self)
		self.HAHAlabel = QtGui.QLabel(str(self.predictedReactionsAbsoluteHAHA)+" (norm.: "+str(self.predictedReactionsNormalizedHAHA)+")",self)
		self.HAHAlabel.resize(100, 50)
		self.HAHAlabel.move(105,110)

		self.WOWimage = QtGui.QPixmap('Images/wow.png')
		self.WOWimage = self.WOWimage.scaledToWidth(20)
		self.WOWlabelimage = QtGui.QLabel(self)
		self.WOWlabelimage.resize(40, 40)
		self.WOWlabelimage.move(80,140)
		self.WOWlabelimage.setPixmap(self.WOWimage)
		# self.WOWlabel = QtGui.QLabel(str(self.predictedReactionsAbsolute[3])+" ("+str(self.predictedReactionsPercentage[3])+"%)",self)
		self.WOWlabel = QtGui.QLabel(str(self.predictedReactionsAbsoluteWOW)+" (norm.: "+str(self.predictedReactionsNormalizedWOW)+")",self)
		self.WOWlabel.resize(100, 50)
		self.WOWlabel.move(105,135)

		self.SADimage = QtGui.QPixmap('Images/sad.png')
		self.SADimage = self.SADimage.scaledToWidth(20)
		self.SADlabelimage = QtGui.QLabel(self)
		self.SADlabelimage.resize(40, 40)
		self.SADlabelimage.move(80,165)
		self.SADlabelimage.setPixmap(self.SADimage)
		# self.SADlabel = QtGui.QLabel(str(self.predictedReactionsAbsolute[4])+" ("+str(self.predictedReactionsPercentage[4])+"%)",self)
		self.SADlabel = QtGui.QLabel(str(self.predictedReactionsAbsoluteSAD)+" (norm.: "+str(self.predictedReactionsNormalizedSAD)+")",self)
		self.SADlabel.resize(100, 50)
		self.SADlabel.move(105,160)

		self.ANGRYimage = QtGui.QPixmap('Images/angry.png')
		self.ANGRYimage = self.ANGRYimage.scaledToWidth(20)
		self.ANGRYlabelimage = QtGui.QLabel(self)
		self.ANGRYlabelimage.resize(40, 40)
		self.ANGRYlabelimage.move(80,190)
		self.ANGRYlabelimage.setPixmap(self.ANGRYimage)
		# self.ANGRYlabel = QtGui.QLabel(str(self.predictedReactionsAbsolute[5])+" ("+str(self.predictedReactionsPercentage[5])+"%)",self)
		self.ANGRYlabel = QtGui.QLabel(str(self.predictedReactionsAbsoluteANGRY)+" (norm.: "+str(self.predictedReactionsNormalizedANGRY)+")",self)
		self.ANGRYlabel.resize(100, 50)
		self.ANGRYlabel.move(105,185)

		self.entropyLabel = QtGui.QLabel("Entropy: "+ str(self.entropy),self)
		self.entropyLabel.resize(200, 50)
		self.entropyLabel.setWordWrap(True)
		self.entropyLabel.move(55,215)

		self.controversialLabel = QtGui.QLabel("Controversial level: "+ self.controversialLevel,self)
		self.controversialLabel.resize(200, 50)
		self.controversialLabel.setWordWrap(True)
		self.controversialLabel.move(55,245)

		self.restart = QtGui.QPushButton("Calculate new post", self)
		self.restart.resize(200, 50)
		self.restart.move(55,340)
		self.restart.clicked.connect(self.restartNewPost)
		
	def restartNewPost(self):
		self.parent.toStartScreen()
		self.close()





def main():
	app = QtGui.QApplication(sys.argv)

	window = Window()
	sys.exit(app.exec_())



if __name__ == '__main__':
	main()