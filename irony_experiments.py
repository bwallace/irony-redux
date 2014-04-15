"""
run experiments for irony classification.
"""

# system imports
import pdb
import sqlite3
import sys
import collections
from collections import defaultdict
import re
import itertools
import random

# external
import nltk # for metrics
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import cross_validation
from sklearn.preprocessing import normalize

import scipy
import numpy as np
import statsmodels.api as sm
import configparser 

# custom kit
from interaction_term_vectorizer import InteractionTermCountVectorizer
# this provides an interface of sorts to the database; 
# **all talking to the db should be done through this 
# module**
# @TODO clean up annotation_stats, which is kind of a mess.
import annotation_stats as db_helper

# just a helper "sign" function
sgn = lambda x : [1 if x_i > 0 else -1 for x_i in x]

def sentence_classification(model="SVC", 
                            add_interactions=False, add_thread_level_interactions=False,
                            verbose=False, tfidf=True, max_features=50000,
                            n_folds=5, seed=30, add_sentiment=False):
    
    if add_thread_level_interactions and not add_interactions:
        raise Exception, "cannot add thread-level interactions with baseline interactions"

    print "-- sentence classification! ---"
    ####
    # get comments, figure out which subreddits they're from
    ####

    # all comments labeled by three people
    labeled_comment_ids = db_helper.get_labeled_thrice_comments()
    # divvy labeled comments up according to whether they are 
    # from the conservative or progressive subreddit
    labeled_conservative_comment_ids = list(set(_keep_ids_in_list(
                db_helper.get_all_comments_from_subreddit("Conservative"), labeled_comment_ids)))
    labeled_progressive_comment_ids = list(set(_keep_ids_in_list(
                db_helper.get_all_comments_from_subreddit("progressive"), labeled_comment_ids)))
    # 'all' here implicitly means all labeled
    all_comment_ids = labeled_conservative_comment_ids + labeled_progressive_comment_ids

    print "%s political comments are labeled thrice" % len(all_comment_ids)
    # create maps from sentence ids to subreddits, and from comment
    # ids to corresponding sentences ids (ie., those sentences comprising it)
    sentence_ids_to_subreddits, comments_d, all_sentence_ids = \
                                _map_sentences_to_subreddits(all_comment_ids)

    # this defines how we map from multiple labels (3) to a single
    # summary label
    collapse_f = lambda lbl_set: 1 if lbl_set.count(1) >= 2 else -1

    all_sentence_ids, sentence_texts, sentence_lbls = \
        db_helper.get_texts_and_labels_for_sentences(all_sentence_ids, 
                                        repeat=False, collapse=collapse_f)

    ####
    # set up some convenient dictionaries 
    sentence_ids_to_parses = dict(zip(all_sentence_ids, db_helper.get_parses(all_sentence_ids)))
    sentence_ids_to_sentiments = dict(zip(all_sentence_ids, db_helper.get_sentiments(all_sentence_ids)))
    sentence_ids_to_labels = dict(zip(all_sentence_ids, sentence_lbls))
    sentence_ids_to_rows = dict(zip(all_sentence_ids, range(len(all_sentence_ids))))

    if add_interactions:
        vectorizer = InteractionTermCountVectorizer(
                            ngram_range=(1,2), stop_words="english", binary=False, 
                            max_features=max_features)
        ###
        # figure out sentence indices corresponding to progressive comments
        progressive_indices, conservative_indices = [], []
        for i in xrange(len(all_sentence_ids)):
            s_id = all_sentence_ids[i]
            subreddit_i = sentence_ids_to_subreddits[s_id]
            if subreddit_i == "progressive":
                progressive_indices.append(i)
            else:
                conservative_indices.append(i)

        all_doc_features, NNPs = _make_interaction_features(
                all_sentence_ids, sentence_ids_to_parses,
                sentence_ids_to_rows, sentence_texts, 
                sentence_ids_to_sentiments, 
                add_thread_level_interactions=add_thread_level_interactions,
                sentence_ids_to_subreddits=sentence_ids_to_subreddits,
                add_sentiment=False)

        X = vectorizer.fit_transform(sentence_texts, 
                interaction_prefixes=["conservative-NNP", "progressive-NNP"],#interaction_prefixes=["progressive", "NNP"],
                interaction_doc_indices=[conservative_indices, progressive_indices],#, progressive_indices],
                interaction_terms=[NNPs, NNPs],
                singleton_doc_features=all_doc_features)

    else:
        vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english", binary=False, 
                                        max_features=max_features)
        X = vectorizer.fit_transform(sentence_texts)



    if add_sentiment:
        X0 = scipy.sparse.csr.csr_matrix(np.zeros((X.shape[0], 2)))
        X = scipy.sparse.hstack((X, X0)).tocsr()
        for i in xrange(X.shape[0]):
            sentence_id = all_sentence_ids[i]
            X[i, X.shape[1] - 1] = 1 if sentence_ids_to_sentiments[sentence_id] <= 0 else -1
            X[i, X.shape[1] - 2] = db_helper.get_sentiment_discrepancy(sentence_id, sentence_ids_to_sentiments) 

    # row normalize features. Make sure that you are not normalizing twice. 
    # Commnet out lines 993 and 994 in sklearn/feature_extract/text.py. 
    # Or add features and run tfidf afterwards and get rid of this if-clause.
    # if tfidf:
    #     X = normalize(X, norm='l2', copy=False)
    if tfidf:
        transformer = TfidfTransformer()
        X = transformer.fit_transform(X)

    #### 
    # sentiment magic!
    if False:
    #if add_sentiment:
        #X0 = scipy.sparse.csr.csr_matrix(np.zeros((X.shape[0], 2)))
        total_sent_features = 5 + 7
        X0 = scipy.sparse.csr.csr_matrix(np.zeros((X.shape[0], total_sent_features)))
        X = scipy.sparse.hstack((X, X0)).tocsr()
        sentiment_col, descrep_col = X.shape[1] - 1, X.shape[1] - 2
        #sent_j = X.shape[1] - total_sent_features - 1

        for i in xrange(X.shape[0]):
            sentence_id = all_sentence_ids[i]
            ## this can be -2 to 2
            #cur_sent_j = sent_j + sentence_ids_to_sentiments[sentence_id]+2
            #X[i, cur_sent_j] = 1.0
            
            X[i, sentiment_col] = sentence_ids_to_sentiments[sentence_id]
            #descrep = db_helper.get_sentiment_discrepancy(
            #                 sentence_id, sentence_ids_to_sentiments)  
            #cur_sent_j = sent_j + 5 + (descrep + 3)
            #X[i, cur_sent_j] = 1.0

            X[i, descrep_col] = db_helper.get_sentiment_discrepancy(
                        sentence_id, sentence_ids_to_sentiments)  



    ####
    # ok -- now we cross-fold validate
    ####
    kf = KFold(len(all_comment_ids), n_folds=n_folds, shuffle=True, random_state=seed)

    recalls, precisions, Fs, AUCs = [], [], [], []
    for train, test in kf:
        train_comment_ids = db_helper._get_entries(all_comment_ids, train)
        train_rows, y_train = _get_rows_and_y_for_comments(train_comment_ids, 
                                comments_d, sentence_ids_to_rows, sentence_ids_to_labels)

        test_comment_ids = db_helper._get_entries(all_comment_ids, test)
        test_rows, y_test = _get_rows_and_y_for_comments(test_comment_ids, 
                                comments_d, sentence_ids_to_rows, sentence_ids_to_labels)
        X_train, X_test = X[train_rows], X[test_rows]

        if model == "SVC":
            svc = LinearSVC(loss="l2", penalty="l2", dual=False, class_weight="auto")
            parameters = {'C':[ .0001, .001, .01,  .1, 1, 10, 100]}
            clf = GridSearchCV(svc, parameters, scoring='f1')
            clf.fit(X_train, y_train)
            preds = sgn(clf.decision_function(X_test))
        elif model == "baseline":
            # guess at chance
            p_train = len([y_i for y_i in y_train if y_i > 0])/float(len(y_train))
            def baseline_clf(): # no input!
                if random.random() < p_train:
                    return 1
                return -1
            preds = [baseline_clf() for i in xrange(len(y_test))]

        if verbose:
            print db_helper.show_most_informative_features(vectorizer, clf.best_estimator_)

        print sklearn.metrics.classification_report(y_test, preds)
        prec, recall, f, support = sklearn.metrics.precision_recall_fscore_support(
                                    y_test, preds, beta=1)
        recalls.append(recall)
        precisions.append(prec)
        Fs.append(f)

    avg = lambda l : sum(l)/float(len(l))
    print "-"*20 + " summary " + "-"*20
    print "model: %s" % model
    print "add interactions?: %s" % add_interactions
    print "add *thread* level interactions?: %s" % add_thread_level_interactions
    print "add sentiment?: %s" % add_sentiment
    print "-"*20 + "results" + "-"*20
    print "average F: %s \naverage recall: %s \naverage precision: %s " % (
                avg(Fs)[1], avg(recalls)[1], avg(precisions)[1])
    print "-"*49
    return Fs, recalls, precisions


def _get_rows_and_y_for_comments(comment_ids, comments_d, sentence_ids_to_rows, 
                                    sentence_ids_to_labels):
    rows, y = [], []
    for comment in comment_ids:
        sentence_ids = comments_d[comment]["sentence_ids"]
        for sent_id in sentence_ids:
            rows.append(sentence_ids_to_rows[sent_id])
            y.append(sentence_ids_to_labels[sent_id])
    return rows,y

def _make_interaction_features(all_sentence_ids, sentence_ids_to_parses, 
                                sentence_ids_to_rows, sentence_texts, 
                                sentence_ids_to_sentiments, 
                                add_thread_level_interactions=False, 
                                sentence_ids_to_subreddits=None,
                                add_sentiment=True):
        
    NNPs = []
    ###
    # note the add_comment_NNPs should really be changed to 'add_thread_NNPs'
    sentence_ids_to_NNPs = db_helper._get_NNP_tokens(all_sentence_ids, sentence_ids_to_parses, 
                                            combine_adjacent=False,
                                            add_comment_NNPs=add_thread_level_interactions)
    
   
    # slightly confusingly, documents==sentences here,
    # (from the perspective of the vectorizer, these are documents)
    all_doc_features_d = {}
    for sent_id, sent_NNPs in sentence_ids_to_NNPs.items():
        cur_doc_features = []

        ### @TODO not sure if we want to add this feature here,
        # DK has done some additional engineering....
        if add_sentiment:
            if sentence_ids_to_sentiments[sent_id] > 0:
                cur_doc_features.append("sentiment-positive")

            if sentence_ids_to_sentiments[sent_id] > 1:
                cur_doc_features.append("sentence-REALLY-positive")

        for sent_NNP in sent_NNPs:
            s = sent_NNP.lower()
            
            index_ = sentence_ids_to_rows[sent_id]
            sent_text = sentence_texts[index_]
            # note that the first check here is redundant, because
            # if we're not including thread features, we would not
            # see NNP's that are not in the comment text. but 
            # better to be explicit
            if add_thread_level_interactions and sent_NNP not in sent_text:
                # then this NNP was extracted from the reddit *thread*
                # not the actual *comment* -- here we add interaction features
                # that cross NNP's in threads with 
                if sentence_ids_to_sentiments[sent_id] > 0:
                    cur_doc_features.append("comment-%s-%s-positive" % 
                                    (sentence_ids_to_subreddits[sent_id], s))      
                
            else:
                NNPs.append(s)
            #pdb.set_trace()
        all_doc_features_d[sent_id] = cur_doc_features

    NNPs = list(set(NNPs))
    all_doc_features = []
    for id_ in all_sentence_ids:
        all_doc_features.append(all_doc_features_d[id_])

    return all_doc_features, NNPs

def _map_sentences_to_subreddits(all_comment_ids):
    sent_ids_to_subreddits = {}
    comments_d = {} # point from comment_id to sentences, etc.
    all_sentence_ids = [] # keep a flat list of sentence ids, too
    for comment_id in all_comment_ids:
        comment_sentence_ids, comment_subreddit = db_helper.get_sentence_ids_for_comment(comment_id)
        comments_d[comment_id] = {"sentence_ids":comment_sentence_ids, 
                                    "subreddit":comment_subreddit}
        for sent_id in comment_sentence_ids:
            sent_ids_to_subreddits[sent_id] = comment_subreddit
            all_sentence_ids.append(sent_id)
    return sent_ids_to_subreddits, comments_d, all_sentence_ids

def _keep_ids_in_list(ids, target_list):
    """ return a filtered version of ids that excludes
        those entries that are *not* found in the target_list """
    return [id_ for id_ in ids if id_ in target_list]




