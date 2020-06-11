# -*- coding: UTF-8 -*-
import re

class CorpusParser:
	def __init__(self, filename):
		self.filename = filename
		self.regex = re.compile('^#\s*\d+')
		self.corpus = dict()
		self.docid = 0

	def parse(self):
		with open(self.filename, encoding='utf-8') as f:
			lines = f.readlines()

		for x in lines:
			self.corpus[self.docid] = x.strip()
			self.docid +=1
	def get_corpus(self):
		return self.corpus


class QueryParser:
	def __init__(self, filename):
		self.filename = filename
		self.queries = []

	def parse(self):
		lines = []
		with open(self.filename, encoding='utf-8') as f:
			lines = f.readlines()
		self.queries = [x.strip() for x in lines]
		

	def get_queries(self):
		return self.queries


if __name__ == '__main__':
	qp = QueryParser('text/queries.txt')
	print (qp.get_queries())