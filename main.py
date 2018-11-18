import os, sys, re
from pyspark import SparkConf, SparkContext
import numpy as np
from numpy import array, argmax
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

conf = SparkConf()

conf.set("spark.executor.memory", "6G")
conf.set("spark.driver.memory", "8G")
conf.set("spark.executor.cores", "4")

conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
conf.set("spark.kryoserializer.buffer.max", "512")
conf.set("spark.default.parallelism", "4")
conf.setMaster('local[4]')

sc = SparkContext(conf=conf)
label_encoder = LabelEncoder()

def integer_encode(label_encoder, np_arr):
    return label_encoder.fit_transform(np_arr)

train_file = sc.textFile('./train-1.dat')
test_file = sc.textFile('./test-1.dat')

lines = train_file.map(lambda f: f.split('\n'))

# document_tuple is in the form of (n, '...'), where n is the class of the document
document = lines.map(lambda line: line[0].split('\t'))

document_class_list = document.map(lambda d: d[0]).collect()
document_words_list = document.map(lambda d: d[1])\
                    .map(lambda str: str.split(' ')).collect()

for document in document_words_list:
    document_words_list.append(integer_encode(label_encoder, array(document)))
    document_words_list.pop(0)

# document_words_2D = np.zeros([len(document_words_list), len(max(document_words_list, key = lambda l: len(l)))], dtype=object)

# for i, j in enumerate(document_words_list):
#     document_words_2D[i][0:len(j)] = j

# classifier = MultinomialNB()
# classifier.fit(document_words_2D, document_class_list)

# test_document = sc.textFile('./test-1.dat')

# test_lines = test_document.map(lambda document: document.split('\n'))
# test_words = test_lines.map(lambda line: line[0].split(' '))

# for e in test_words.collect():
#     print(classifier.predict(e))



