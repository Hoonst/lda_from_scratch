import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import random
import pickle

class LDA:
    def __init__(self, data, n_samples = 1000, T = 10, iterate = 10, n_top_words = 10):
        self.data = data
        self.n_samples= n_samples
        self.T = T
        self.iterate = iterate
        self.n_top_words = n_top_words
        print("Initialize LDA with Data is O.N")


    def preprocess(self, n_samples):
        # n_samples = 1000
        data_samples = self.data[:n_samples]

        # 데이터 내에 존재하는 단어에 대한 빈도(Count)를 계산하면 용량이 매우 커지기 때문에
        # Sparse Matrix를 효율적으로 보존하는 CountVectorizer 사용
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=10000,
                                        stop_words='english')
        tf = tf_vectorizer.fit_transform(data_samples)

        # 단어에 대한 index 설정
        self.vocabulary = tf_vectorizer.vocabulary_

        self.documents = []

        # tf.toarray()에 담겨있는 문서들을 하나씩 순회하면서
        for row in tf.toarray():

            # count가 0이 아닌 index를 파악 > present_words
            present_words = np.where(row != 0)[0].tolist()
            present_words_with_count = []

            # present_words에 담겨 있는 index(count가 0이 아닌 index)에 대하여
            for word_idx in present_words:
                # 실제 count를 구하고 count만큼 index를 담는다.
                for count in range(row[word_idx]):
                    present_words_with_count.append(word_idx)

            self.documents.append(present_words_with_count)

        self.D = len(self.documents)
        self.V = len(self.vocabulary)
        print("Document for LDA is O.N")

    def parameter_initialization(self, T = 10):
        # document 하나씩 길이에 따라서 단어들의 Topic을 반영할 준비를 word_topic_in_document에 해둔다.
        self.word_topic_in_document = [[0 for _ in range(len(document))] for document in self.documents]  # z_i_j

        self.T = T
        self.alpha = 1 / self.T
        self.beta = 1/ self.T
        self.document_topic_dist = np.zeros((self.D, self.T))   # 문서 내의 Topic Distribution
        self.topic_word_dist = np.zeros((self.T, self.V))       # 토픽 내의 Word Distribution
        self.document_words_cnt = np.zeros((self.D))       # 전체 Document의 단어 갯수
        self.topic_words_cnt = np.zeros((self.T))          # 전체 Topic의 단어 갯수

        for document_index, document in enumerate(self.documents):
            # 모든 문서 내의 단어들을 하나씩 순회하면서
            for word_index, word in enumerate(document):
                # 일단 Random Function을 사용해 Topic 갯수로 지정한 T개만큼의 Topic을 Random 배정
                self.word_topic_in_document[document_index][word_index] = random.randint(0,T-1)

                # 배정한 Word_topic
                word_topic = self.word_topic_in_document[document_index][word_index]

                # document 내의 topic 분포를 알기 위하여, 배정된 Topic을 하나씩 더한다.
                self.document_topic_dist[document_index][word_topic] += 1
                self.topic_word_dist[word_topic, word] += 1

                self.document_words_cnt[document_index] += 1
                self.topic_words_cnt[word_topic] += 1

        print("Parameter Initialization O.N")

    def gibbs_sampling(self, iterate = 10):
        self.iterate = iterate
        for iteration in tqdm(range(self.iterate)):
            for document_index, document in enumerate(self.documents):
                for word_index, word in enumerate(document):
                    word_topic = self.word_topic_in_document[document_index][word_index]

                    # 해당 단어를 모든 분포 내에서 하나씩 임시로 뺀다.
                    self.document_topic_dist[document_index][word_topic] -= 1
                    self.topic_word_dist[word_topic, word] -= 1
                    self.topic_words_cnt[word_topic] -= 1

                    # Update Process: 새로운 Topic을 단어에 반영하는 절차
                    document_topic_expectation= (self.document_topic_dist[document_index] + self.alpha) / (self.document_words_cnt[document_index] - 1 + self.T * self.alpha)
                    topic_word_expectation = (self.topic_word_dist[:, word] + self.beta) / (self.topic_words_cnt + self.V * self.beta)
                    new_topic_dist = document_topic_expectation * topic_word_expectation
                    new_topic_dist /= np.sum(new_topic_dist)

                    # 새롭게 구성된 분포에서 확률이 높은 값의 index
                    new_topic = np.random.multinomial(1, new_topic_dist).argmax()

                    self.word_topic_in_document[document_index][word_index] = new_topic
                    self.document_topic_dist[document_index][new_topic] += 1
                    self.topic_word_dist[new_topic, word] += 1
                    self.topic_words_cnt[new_topic] += 1

        print("Gibbs Sampling O.N")

    def show_topics(self, n_top_words = 10):
        index_vocabulary = {v: k for k, v in self.vocabulary.items()}
        self.n_top_words = n_top_words

        for topic_idx, topic in enumerate(self.topic_word_dist):
            message = "Topic #%d: " % topic_idx
            message += " ".join([index_vocabulary[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)

    def lda_process(self):
        self.preprocess(self.n_samples)
        self.parameter_initialization(self.T)
        print("="*50)
        print("Current Parameter List")
        print(f'n_samples: {self.n_samples} / Topics: {self.T} / iteration: {self.iterate} / n_top_words: {self.n_top_words}')
        self.gibbs_sampling(self.iterate)
        self.show_topics(self.n_top_words)
