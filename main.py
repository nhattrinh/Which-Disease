import os, sys, re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

def main():
    test_docs = []
    train_docs = []
    train_classes = []

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer(use_idf=False)

    with open('train.dat','r') as infile:
        data_split = [line.split('\t') for line in infile.read().splitlines()]
        for d in data_split:
            train_docs.append(d[1])
            train_classes.append(d[0])
        infile.close()

    with open('test.dat','r') as infile:
        test_docs.extend(infile.read().splitlines())
        infile.close()

    train_counts = vectorizer.fit_transform(train_docs,None)
    test_counts = vectorizer.transform(test_docs)

    train_tfidf = transformer.fit_transform(train_counts, None)
    test_tfidf = transformer.transform(test_counts)

    classifier = SGDClassifier(tol=None, max_iter=2000).fit(train_tfidf, train_classes)

    predictions = classifier.predict(test_tfidf)

    with open('format.dat','w') as outfile:
        outfile.write('\n'.join(predictions))
        outfile.close()

if __name__ == '__main__':
    main()
