from invdx import build_data_structures
from rank import score_BM25
import operator


class QueryProcessor:
	def __init__(self, queries, corpus):
		self.queries = queries
		self.index, self.dlt = build_data_structures(corpus) #index: word frequence in per document corpus[word][docid]

	def run(self):
		results = []
		for query in self.queries:
			results.append(self.run_query(query))
			print(self.run_query(query))
		return results

	def run_query(self, query):
		query_result = dict()
		for init_docid in range(1951):
			query_result[init_docid] = 0
		for term in query:
			if term in self.index:
				doc_dict = self.index[term] # retrieve index entry
				for key in doc_dict.keys():
					docid, freq = int(key), doc_dict[key]
					score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=len(self.dlt),
									   dl=self.dlt.get_length(docid), avdl=self.dlt.get_average_length()) # calculate score
					if docid in query_result: #this document has already been scored once
						query_result[docid] += score
					else:
						query_result[docid] = score
		return query_result