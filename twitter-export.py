#!/usr/bin/python

# https://twitter.com/statuses/user_timeline/albertzeyer.xml?page=x

import better_exchook
better_exchook.install()

twitterUser = "albertzeyer"
# TODO: maybe allow other twitter users...

import time, os, sys
import BeautifulSoup
from urllib2 import Request, HTTPError, URLError, urlopen

def getXml(url):
	while True:
		try:
			req = Request(url)
			open_req = urlopen(req)
			content = open_req.read()
			content_type = open_req.headers.get('content-type')
			break
		except HTTPError, e:
			print e
			if e.code in (502,503): # bad gateway, Service Unavailable. happens if overloaded
				print "waiting a few seconds"
				time.sleep(2)
				continue
			if e.code == 400 and int(e.hdrs["X-RateLimit-Remaining"]) == 0:
				print "rate limit exceeded. waiting 1h"
				time.sleep(60 * 60)
				continue
			raise e
		assert False
	assert "xml" in content_type
	soup = BeautifulSoup.BeautifulSoup(content)
	return soup

mydir = os.path.dirname(__file__)
LogFile = mydir + "/twitter.log"
print "logfile:", LogFile

try:
	log = eval(open(LogFile).read())
	assert isinstance(log, dict)
except IOError: # e.g. file-not-found. that's ok
	log = {}
except:
	print "logfile reading error"
	sys.excepthook(*sys.exc_info())
	log = {}

def betterRepr(o):
	# the main difference: this one is deterministic
	# the orig dict.__repr__ has the order undefined.
	if isinstance(o, list):
		return "[" + ", ".join(map(betterRepr, o)) + "]"
	if isinstance(o, tuple):
		return "(" + ", ".join(map(betterRepr, o)) + ")"
	if isinstance(o, dict):
		return "{\n" + "".join(map(lambda (k,v): betterRepr(k) + ": " + betterRepr(v) + ",\n", sorted(o.iteritems()))) + "}"
	# fallback
	return repr(o)
	
def saveLog():
	global log, LogFile
	f = open(LogFile, "w")
	f.write(betterRepr(log))
	f.write("\n")

def formatDate(t):
	return time.strftime("%Y-%m-%d %H:%M:%S", t)
	
# log is dict: (date, id) -> tweet, date as in formatDate

def updateTweetFromSource(tweet, s):
	tweet["text"] = s.find("text").text
	tweetGeo = s.find("georss:polygon")
	tweet["geo"] = tweetGeo.text if tweetGeo else None
	if s.in_reply_to_status_id.text:
		retweetFrom = tweet.setdefault("retweeted-from", {})
		retweetFrom["status-id"] = long(s.in_reply_to_status_id.text)
		retweetFrom["user-id"] = long(s.in_reply_to_user_id.text)
		retweetFrom["user-name"] = s.in_reply_to_screen_name.text

def updateTweet(tweet):
	pass

SkipOldWebupdate = False
DataCount = 1000

from itertools import *
for pageNum in count(1):
	print "> page", pageNum
	data = getXml("https://twitter.com/statuses/user_timeline/%s.xml?page=%i&count=%i" % (twitterUser, pageNum, DataCount))

	statuses = []	
	for s in data.statuses.childGenerator():
		if isinstance(s, (str,unicode)): continue
		assert isinstance(s, BeautifulSoup.Tag)
		assert s.name == "status"
		statuses += [s]

	if not statuses:
		print "** finished"
		break
		
	for s in statuses:
		tweetId = long(s.id.text)
		tweetDate = formatDate(time.strptime(s.created_at.text, "%a %b %d %H:%M:%S +0000 %Y"))
		tweetKey = (tweetDate, tweetId)
		if SkipOldWebupdate and tweetKey in log:
			print "** hit old entry, finished"
			pageNum = None
			break
		tweet = log.setdefault(tweetKey, {})
		tweet["id"] = tweetId
		tweet["date"] = tweetDate
		updateTweetFromSource(tweet, s)
		updateTweet(tweet)
		saveLog()
	if not pageNum: break

for tweet in log.itervalues():
	updateTweet(tweet)
saveLog()
