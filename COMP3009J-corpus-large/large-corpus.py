import sys
import os
import json
import string
import re

# sys.path.append('./files')
from files import porter
from math import log2
import time

class Large_Corpus():

    def __init__(self):

        '''
        document_path: document path
        document_num: number of documents
        '''
        self.document_path = "./documents"
        self.document_num = 0

        ''' 
        judgement_path: judgement path
        judge_dic: the dictionary to store the judged documents
            key: query name
            value: {key: document name, value: relevant score}
        '''
        self.judgement_path = "./files/qrels.txt"
        self.judgement_dic = {}

        '''
        rel_judgement: dictionary that stores number of relevent documents for all query
            key: query name
            value: number of relevent documents
        '''
        self.rel_judgement = {}

        '''
        query_path: query path
        '''
        self.query_path = "./files/queries.txt"
        
        '''
        stoprword_path: stopword path
        stopword_set: the set to store the stopword
        '''
        self.stopword_path = "./files/stopwords.txt"
        self.stopword_set = set()

        '''
        stemmer: stemmer
        stemming_dictionary: the dictionary to store the words have pasted the stemming
            key: word before stemming
            value: word after stemming
        '''
        self.stemmer = porter.PorterStemmer()
        self.stemming_dictionary = {}
        
        ''' 
        term_index: index for number of terms for all documents
            key: document name
            value: number of terms (document length)
        term_number: the sum of the term number
        '''
        self.term_index = {}
        self.term_num = 0
        
        '''
        index_path: index path
        index: index for terms in the corpus
            key: term
            value: {key: document name, value: number of terms}
        '''
        self.index_path = "./index.json"
        self.index = {}

        '''
        k, b: parameters for BM25 calculation
        avg_length: average document length
        '''
        self.k = 1
        self.b = 0.75
        self.avg_length = 0
        
        '''
        calculation_dic: temporary BM25 calculation dictionary
            key: term in query
            value: pre-calculation result
        '''
        self.calculation_dic = {}

        '''
        score_dic: BM25 result dictionary
            key: query name
            value: {key: document name, value: score}
        '''
        self.score_dic = {}

        '''
        result_list: list of returned document name for each query
        ''' 
        self.result_list = []

        '''
        the sdum of the score for 7 evaluations
        '''
        self.percision = 0
        self.recall = 0
        self.p_10 = 0
        self.r_percision = 0
        self.ap = 0
        self.bpref = 0
        self.ndcg = 0

        '''
        output_path: output path
        '''
        self.output_path = "./output.txt"


    def check_index(self):
        '''
        check if the index file exists
        return True if exists
        return False if not exists
        '''
        if os.path.exists(self.index_path):
            return True
        else:
            return False


    def load_index(self):
        '''
        load two index from the list in external file and store them separety
        '''
        with open(self.index_path, 'r', encoding='UTF-8') as i:
            dicts = json.load(i)
            self.index = dicts[0]
            self.term_index = dicts[1]


    def create_stopword_set(self):
        '''
        load stopwords from the external file and store them
        '''
        with open(self.stopword_path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                self.stopword_set.add(line.strip())


    def create_index(self):
        '''
        create the two index and store them
        '''
        # get the document directory list
        document_dir_list = os.listdir(self.document_path)
        if '.DS_Store' in document_dir_list:
            document_dir_list.remove('.DS_Store')

        for dir in document_dir_list:
             # get the document name list
            document_path_list = os.listdir(self.document_path + "/" + dir)
            for filename in document_path_list:
                # open each document
                with open(os.path.join(self.document_path + "/" + dir, filename), 'r', encoding='UTF-8') as f:
                    lines = f.readlines()
                    # prepare to count the number of terms for each document
                    term_num = 0
                    for line in lines:
                        # implement tokenisation for each line
                        token_list = self.tokenisation(line)
                        for token in token_list:
                            # check if the token is a stopword
                            if token not in self.stopword_set:
                                # implement stemming for each token
                                if token in self.stemming_dictionary:
                                    # if the token has been stemmed, get the result directly
                                    term = self.stemming_dictionary[token]
                                else:
                                    # if the token has not been stemmed, stem it and store the result
                                    term = self.stemmer.stem(token)
                                    self.stemming_dictionary[token] = term
                                    # if the term hasn't appeared in the index, create a dictionary for it
                                    if term not in self.index:
                                        self.index[term] = {}
                                # if the document hasn't appeared in the term dictionary, add it
                                if filename not in self.index[term]:
                                    self.index[term][filename] = 0
                                # increase the number of the current term appeared in the current document
                                self.index[term][filename] += 1
                                term_num += 1
                    # store the document length
                    self.term_index[filename] = term_num


    def tokenisation(self, line):
        '''
        implement tokenisation to the line input and return a list of token
        '''
        # replace all punctuations except hyphen to space
        non_punct = re.sub('([^\u0030-\u0039\u0041-\u005a\u0061-\u007a\u002d\u0020])', ' ', line)
        # transform the line to all lower-case
        non_upper = non_punct.lower()
        # split the line using the space
        token_list = non_upper.strip().split()
        return token_list


    def store_index(self):
        '''
        append thwe two index to a list and store it in an external file
        '''
        index_list = [self.index, self.term_index]
        with open(self.index_path, 'a', encoding='UTF-8') as i:
            json.dump(index_list, i)


    def pre_calculation(self):
        '''
        calculate the number of document and the average document length
        '''
        # get the number of documents
        self.document_num = len(self.term_index)
        # calculate the sum of the length of documents
        for term in self.term_index.values():
            self.term_num += term 
        # calculate the average document length
        self.avg_length = self.term_num / self.document_num


    def load_query(self, query):
        '''
        load queries from input and create a temporary dictionary for further calculation
        '''
        # implement tokenisation
        token_list = self.tokenisation(query)
        for token in token_list:
            if token not in self.stopword_set:
                # implement stemming
                if token in self.stemming_dictionary:
                    term = self.stemming_dictionary[token]
                else:
                    term = self.stemmer.stem(token)
                    self.stemming_dictionary[token] = term
                # only store the term and the pre-calculated score for term appeared in the corpus
                if term in self.index:
                    # find the term appears in how many documents
                    ni = len(self.index[term])
                    # apply BM25 pre-calculation for the term and store the result
                    self.calculation_dic[term] = (1 + self.k) * log2((self.document_num - ni + 0.5)/(ni + 0.5))


    def calculation(self, query_name, query):
        '''
        perform BM25 calculation and decide the result documents returned
        '''
        # create the dictionary to store the result of the query input
        self.score_dic[query_name] = {}
        for document in self.term_index.keys():
            # calculate score for all documents
            score = 0
            for term in self.calculation_dic.keys():
                if document not in self.index[term]:
                    # if the term doesn't appeared in the current document, pass calculation
                    fij = 0
                else:
                    # find the term appears how many times in the current document
                    fij = self.index[term][document]
                # apply BM25 calculation by using the pre-calculated result
                score += self.calculation_dic[term] * (fij / (fij + self.k*(1 - self.b+((self.b * self.term_index[document])/self.avg_length))))
            # remove the score that is zero and store the valid score
            if score >= 0:
                self.score_dic[query_name][document] = score
        # sort the returned documents by the score and slice the result by introducing the query length
        self.score_dic[query_name] = dict(sorted(self.score_dic[query_name].items(), key=lambda x: x[1], reverse=True)[:round(870/len(query))])
        # clear the pre-calculation dictionary
        self.calculation_dic.clear()


    def display_result(self, query_name):
        '''
        display the result of manual input query
        '''
        print("Results for query [library information conference]")
        # get the top 15 results of both document name and score
        documents = list(self.score_dic[query_name].keys())[:15]
        scores = list(self.score_dic[query_name].values())[:15]
        # print result
        for i in range(len(documents)):
            print(str(i+1) + " " + documents[i] + " " + str(scores[i]))


    def load_judgement(self):
        '''
        load judged documents from the external file and generate the dictionary only contains relevant documents in the judgement'''
        with open(self.judgement_path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                judgement = line.strip().split()
                # if the query hasn't appeared in the dictionary that stores judgement, create a dictionary for it 
                if judgement[0] not in self.judgement_dic:
                    self.judgement_dic[judgement[0]] = {}
                    # set the number of relevant document for the current query to zero
                    self.rel_judgement[judgement[0]] = 0
                # store the relevant score for the current document
                self.judgement_dic[judgement[0]][judgement[2]] = judgement[3]
                # if the document is relevant, increase the number of relevant document
                if judgement[3] != "0":
                    self.rel_judgement[judgement[0]] += 1


    def process_querys(self):
        '''
        load queries from the external file and iterate them to perform BM25 calculation
        '''
        with open(self.query_path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                # split the query and the query name
                query = line.strip().split(" ", 1)
                # load each query and perform calculation
                self.load_query(query[1])
                self.calculation(query[0], query[1])


    def process_evaluations(self):
        '''
        load result of the model and calculate the 7 evaluation metrics
        '''
        # get the names of query
        querys = list(self.score_dic.keys())
        for query in querys:
            # get the result documents for the current query
            self.result_list = list(self.score_dic[query].keys())
            # add each result score to the sum score separetly
            self.percision += self.calculate_precision(query)
            self.recall += self.calculate_recall(query)
            self.p_10 += self.calculate_p_10(query)
            self.r_percision += self.calculate_r_percision(query)
            self.ap += self.calculate_ap(query)
            self.bpref += self.calculate_bpref(query)
            self.ndcg += self.calculate_ndcg(query)


    def calculate_precision(self, query_name):
        '''
        calculate Percision of the current query
        '''
        rel = 0
        # calculate the returned judged relevant document for the query
        for document in self.result_list:
            if document in self.judgement_dic[query_name] and self.judgement_dic[query_name][document] != "0":
                rel += 1
        # claculate Percision
        if len(self.result_list) != 0:
            # number of judged relevant document returned / number of all document returned
            precision = rel / len(self.result_list)
        else:
            precision = 0
        return precision
        
    
    def calculate_recall(self, query_name):
        '''
        calculate Recall of the current query
        '''
        rel = 0
        # calculate the judged relevant document for the query
        for document in self.result_list:
            if document in self.judgement_dic[query_name] and self.judgement_dic[query_name][document] != "0":
                rel += 1
        # calculate Recall
        # number of judged relevant document returned / number of all relevant document
        recall = rel / self.rel_judgement[query_name]
        return recall


    def calculate_p_10(self, query_name):
        '''
        calculate P@10 of the current query
        '''
        rel = 0
        # calculate the judged relevant document from the top 10 returned documents
        for i in range(len(self.result_list[:10])):
            if self.result_list[i] in self.judgement_dic[query_name] and self.judgement_dic[query_name][self.result_list[i]] != "0":
                rel += 1
        # calcaulate P@10
        # number of judged relevant documents from the top 10 returned documents / 10
        p_10 = rel / 10
        return p_10


    def calculate_r_percision(self, query_name):
        '''
        calculate R_precision of the current query
        '''
        rel = 0
        # calculate the judged relevant document from the top relevant numbered returned documents
        for i in range(len(self.result_list[:self.rel_judgement[query_name]])):
            if self.result_list[i] in self.judgement_dic[query_name] and self.judgement_dic[query_name][self.result_list[i]] != "0":
                rel += 1
        # calculate R_percision
        # number of judged relevant document from the top relevant numbered returned documents / number of all relevant document
        r_percision = rel / self.rel_judgement[query_name]
        return r_percision

    
    def calculate_ap(self, query_name):
        '''
        calculate AP of the current query
        '''
        p = 0
        rel = 0
        # calculate the sum of the percision for each judged relevant document
        for i in range(len(self.result_list)):
            if self.result_list[i] in self.judgement_dic[query_name] and self.judgement_dic[query_name][self.result_list[i]] != "0":
                # record number of relevant document
                rel += 1
                p += rel / (i + 1)
        # calculate AP
        # the sum of the percision for each judged relevant document / number of all relevant document
        ap = p / self.rel_judgement[query_name]
        return ap


    def calculate_bpref(self, query_name):
        '''
        calculate bpref of the current query
        '''
        contri = 0
        non_rel = 0
        # calculate the sum of contribution for each judged relevant document
        for document in self.result_list:
            # just check the judged document
            if document in self.judgement_dic[query_name]:
                if self.judgement_dic[query_name][document] != "0":
                    # only calculate contribution before all relevant documents are found
                    if non_rel <= self.rel_judgement[query_name]:
                        contri += 1 - non_rel / self.rel_judgement[query_name]
                else:
                    # record the number of none relevant document
                    non_rel += 1
        # calculate bpref
        # the sum of contribution for each judged relevant document / number of all relevant document
        bpref = contri / self.rel_judgement[query_name]
        return bpref


    def calculate_ndcg(self, query_name):
        '''
        calculate NDCG of the current query
        '''
        # create the ideal returned result 
        ideals = list(dict(sorted(self.judgement_dic[query_name].items(), key=lambda x: x[1], reverse=True)).values())
        dcg = 0
        idcg = 0
        # calculate dcg for the tenth document
        for i in range(len(self.result_list[:10])):
            if self.result_list[i] in self.judgement_dic[query_name] and self.judgement_dic[query_name][self.result_list[i]] != "0":
                if dcg != 0:
                    # find relevant document that is not the first
                    dcg += int(self.judgement_dic[query_name][self.result_list[i]]) / log2(i + 1)
                else:
                    # find the first relevant document
                    dcg += int(self.judgement_dic[query_name][self.result_list[i]])
        # calculate idcg for the tenth document in ideal returned result
        for j in range(len(ideals[:10])):
            if idcg != 0:
                # find relevant document that is not the first
                idcg += int(ideals[j]) / log2(j + 1)
            else:
                # find the first relevant document
                idcg += int(ideals[j])
        # calculate NDCG
        # dcg for the tenth document / idcg for the tenth document in ideal returned result
        ndcg = dcg / idcg
        return ndcg


    def check_output(self):
        '''
        check if the output exists, remove it if exists
        '''
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
            

    def store_output(self):
        '''
        store the result of the model as the output in an external file
        '''
        # get the names of query
        querys = list(self.score_dic.keys())
        with open(self.output_path, 'a', encoding='UTF-8') as f:
            for query in querys:
                # get the results of both document name and score
                documents = list(self.score_dic[query].keys())
                scores = list(self.score_dic[query].values())
                # store result
                for i in range(len(documents)):
                    f.write("".join([query, " Q0 ", documents[i], " ", str(i+1), " ", str(scores[i]), " 19206212\n"]))


    def show_result(self):
        '''
        show the result of the evaluation
        '''
        print("Evaluation results:")
        # calculate the 7 evaluation metrics and print
        print("Percision:      " + str(self.percision / len(self.judgement_dic)))
        print("Recall:         " + str(self.recall / len(self.judgement_dic)))
        print("P@10:           " + str(self.p_10 / len(self.judgement_dic)))
        print("R_percision:    " + str(self.r_percision / len(self.judgement_dic)))
        print("MAP:            " + str(self.ap / len(self.judgement_dic)))
        print("bpref:          " + str(self.bpref / len(self.judgement_dic)))
        print("NDCG:           " + str(self.ndcg / len(self.judgement_dic)))


if __name__ == '__main__':
    '''
    main function
    '''
    t1 = time.time()
    corpus = Large_Corpus()
    corpus.create_stopword_set()
    print("Loading BM25 index from file, please wait.")
    # check if index exists
    if corpus.check_index():
        # index exists, just load
        corpus.load_index()
        t2 = time.time()
        print("The index load time is:", t2-t1, "s")
    else:
        # index not exists, create it
        corpus.create_index()
        corpus.store_index()
        t2 = time.time()
        print("The index store time is:", t2-t1, "s")
    corpus.pre_calculation()

    # prevent run directly
    try:
        # read the mode user chose
        if sys.argv[2] == "manual":
            # manual input mode
            while True:
                query = input("Enter query: ")
                # request input if user hasn't quit
                if query != "QUIT":
                    t3 = time.time()
                    corpus.load_query(query)
                    corpus.calculation("manual", query)
                    corpus.display_result("manual")
                    t4 = time.time()
                    print("The search finished in:", t4-t3, "s")
                else:
                    break
        elif sys.argv[2] == "evaluation":
            # evaluation mode
            t3 = time.time()
            corpus.load_judgement()
            corpus.process_querys()
            corpus.check_output()
            corpus.store_output()
            corpus.process_evaluations()
            corpus.show_result()
            t4 = time.time()
            print("The evaluation finished in:", t4-t3, "s")
    except:
        print("Please use command line to start search")

        