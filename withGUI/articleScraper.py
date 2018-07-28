from bs4 import BeautifulSoup
import requests

def getArticleData(articleURL):
	url = articleURL
	source = url.split("/")[2]

	pageTable = requests.get(url)
	pageTable.content
	soupTable = BeautifulSoup(pageTable.content,'html.parser')

	# get title
	titleBox = soupTable.find('h1')
	title = titleBox.text.strip()

	# get summary
	if "nu" in source:
		summary = soupTable.find('div', class_="item-excerpt").text.strip()
	
	elif "parool" in source:
		summary = soupTable.find('p', class_="article__intro").text.strip()
	
	elif "nrc" in source:
		divs = soupTable.findAll('div', attrs={'class':"intro article__intro"})
		for div in divs:
			summary = div.find("p", text=True).text.strip()
	
	elif "volkskrant" in source:
		summary = soupTable.find('p', class_="artstyle__intro").text.strip()
	
	elif "telegraaf" in source:
		ps = soupTable.findAll('p', attrs={'class':"abril-bold no-top-margin"})
		summary = ""
		for p in ps:
			text = p.findAll("span", text=True)
		for item in text:
			item = item.text.strip()
			summary = summary + " " + item
	
	elif "nos" in source:
		divs = soupTable.findAll('div', attrs={'class':"article_textwrap"})
		for div in divs:
			summary = div.find("p", text=True).text.strip()


	# elif "rtlnieuws" in source:
	# 	pass

	return title, summary







if __name__ == '__main__':
	main()

