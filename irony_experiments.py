"""
run experiments for irony classification.
"""

# system imports
import pdb
import sqlite3
import sys
import os
import collections
from collections import defaultdict
import re
import itertools
import random
import pickle
import operator 

# external
import nltk # for metrics
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve, pairwise, auc 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import cross_validation
from sklearn.preprocessing import normalize

import scipy
import numpy as np
import statsmodels.api as sm
import configparser 

# custom kit
from interaction_term_vectorizer import InteractionTermCountVectorizer

from sklearn.linear_model import stochastic_gradient # standard
from sklearn.linear_model import stochastic_gradient_i # ours
from sklearn.linear_model import stochastic_gradient_i2 # ours ++
# this provides an interface of sorts to the database; 
# **all talking to the db should be done through this 
# module**
# @TODO clean up annotation_stats, which is kind of a mess.
import annotation_stats as db_helper
#import pastcomments_stats as dk

# just a helper "sign" function
sgn = lambda x : [1 if x_i > 0 else -1 for x_i in x]
avg = lambda l : sum(l)/float(len(l))

#max_features=500000,
def sentence_classification_heldout(model="SGD", at_least=2, add_interactions=False, 
                                    add_thread_level_interactions=False, 
                                    add_sentiment=False,
                                    add_interaction_sentiment=False,
                                    tfidf=True, penalty_str="l2",
                                    seed = 10, n_runs=10,
                                    max_features=500000, verbose=False,
                                    fns_out="false-negatives.txt", interaction_baseline=False,
                                    return_ifeature_weights=False):
    train_ids = db_helper.get_labeled_thrice_comments()
    test_ids = db_helper.get_test_comment_ids()
    assert(set(train_ids).intersection(set(test_ids)) == set([]))
    random.seed(seed)
    all_ids = train_ids + test_ids 
    conservative_comment_ids = list(set(_keep_ids_in_list(
                db_helper.get_all_comments_from_subreddit2("Conservative"), all_ids)))
    progressive_comment_ids = list(set(_keep_ids_in_list(
                db_helper.get_all_comments_from_subreddit2("progressive"), all_ids)))
    #pdb.set_trace()
    # i.e., all "political" IDs
    all_ids = conservative_comment_ids + progressive_comment_ids
    # filter train/test ids
    train_ids = [id_ for id_ in train_ids if id_ in all_ids]
    test_ids = [id_ for id_ in test_ids if id_ in all_ids]

    test_conservative_ids = [id_ for id_ in test_ids if id_ in conservative_comment_ids]
    test_conservative_comments = db_helper.grab_comments(test_conservative_ids)
    
    test_progressive_ids = [id_ for id_ in test_ids if id_ in progressive_comment_ids]
    print "%s progressive and %s conservative test comments" % (
                len(test_progressive_ids), len(test_conservative_ids))

    sentence_ids_to_subreddits, comments_d, all_sentence_ids = \
                    _map_sentences_to_subreddits(all_ids)

    if interaction_baseline: 
        for sentence_id in sentence_ids_to_subreddits:
            # dummy
            sentence_ids_to_subreddits[sentence_id] = "interaction-baseline"

    collapse_f = lambda lbl_set: 1 if lbl_set.count(1) >= at_least else -1
    # all labels come from the test_db, because this contains the labels
    # from the development dataset *and* the new labels that were not present
    # therein
    all_sentence_ids, sentence_texts, sentence_lbls = \
        db_helper.get_texts_and_labels_for_sentences(all_sentence_ids, use_test_db=True, 
                                        repeat=False, collapse=collapse_f)

    collapse_f = lambda lbl_set: 1 if lbl_set.count(1) >= 1 else -1
    all_sentence_ids, sentence_texts, sentence_lbls2 = \
        db_helper.get_texts_and_labels_for_sentences(all_sentence_ids, use_test_db=True, 
                                        repeat=False, collapse=collapse_f)

    sentence_ids_to_parses = dict(zip(all_sentence_ids, db_helper.get_parses(all_sentence_ids)))
    sentence_ids_to_sentiments = dict(zip(all_sentence_ids, db_helper.get_sentiments(all_sentence_ids)))
    sentence_ids_to_labels = dict(zip(all_sentence_ids, sentence_lbls))
    sentence_ids_to_labels2 = dict(zip(all_sentence_ids, sentence_lbls2))
    sentence_ids_to_rows = dict(zip(all_sentence_ids, range(len(all_sentence_ids))))
    sentence_ids_to_texts = dict(zip(all_sentence_ids, sentence_texts))


    X, vectorizer = None, None

    if add_interactions or add_sentiment:
        sentiment_only = not add_interactions 
        if sentiment_only:
            print "sentiment only!"
        X, vectorizer = _make_interaction_vectorizer(all_sentence_ids, sentence_texts, 
                                                        sentence_ids_to_subreddits, sentence_ids_to_parses,
                                                        sentence_ids_to_rows, sentence_ids_to_sentiments, 
                                                        add_thread_level_interactions=add_thread_level_interactions,
                                                        max_features=max_features,
                                                        add_sentiment=add_sentiment, 
                                                        add_interaction_sentiment=add_interaction_sentiment,
                                                        sentiment_only=sentiment_only)
    else:
        vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english", binary=False, 
                                        max_features=max_features)
        X = vectorizer.fit_transform(sentence_texts)

    if tfidf:
        transformer = TfidfTransformer()
        X = transformer.fit_transform(X)

    print "number of runs: %s" % n_runs

    recalls, precisions, Fs, accuracies = [], [], [], []
    ifeatures = defaultdict(list)
    for run in xrange(n_runs):  
        cur_seed = seed + (run + 1)

        ### ok, showtime
        train_rows, train_sentences, y_train = _get_rows_and_y_for_comments(train_ids, 
                                comments_d, sentence_ids_to_rows, sentence_ids_to_labels)

        test_rows, test_sentences, y_test = _get_rows_and_y_for_comments(test_ids, 
                                comments_d, sentence_ids_to_rows, sentence_ids_to_labels)

        X_train, X_test = X[train_rows], X[test_rows]

        if model == "baseline":
            # guess at chance
            p_train = len([y_i for y_i in y_train if y_i > 0])/float(len(y_train))
            def baseline_clf(): # no input!
                if random.random() < p_train:
                    return 1
                return -1
            preds = [baseline_clf() for i in xrange(len(y_test))]
        
        elif model == "SVC":
            #sgd = stochastic_gradient.SGDClassifier(loss="log", class_weight="auto", 
            #                penalty=penalty_str)
            svc = LinearSVC(loss="l2", penalty="l2", dual=False, class_weight="auto")
            parameters = {'C':[.00001, .0001, .001, .01]}
            clf = GridSearchCV(svc, parameters, scoring='f1')
            clf.fit(X_train, y_train)    
            preds = clf.predict(X_test)
        elif model == "SGD":
            sgd = stochastic_gradient.SGDClassifier(shuffle=True, random_state=cur_seed, 
                                                loss="log", class_weight="auto", penalty=penalty_str)
            parameters = {"alpha":[.005, .001]}
            clf = GridSearchCV(sgd, parameters, scoring='f1')
            #sgd.fit(X_train, y_train)

            clf.fit(X_train, y_train)
        
            preds = clf.predict(X_test)

        elif model == "SGDi":
            penalty_str = "l1l2"     


            b = vectorizer.get_interaction_term_b(iprefixes=["i-"])
            b2 = vectorizer.get_interaction_term_b(iprefixes=["f-"], exclude_containing=["overall-"])

            sgd = stochastic_gradient_i2.SGDClassifier(shuffle=True, random_state=cur_seed, 
                                                        l1_indicator_v1=b, l1_indicator_v2=b2,
                                                        loss="log", penalty=penalty_str, 
                                                        class_weight="auto")
            
            parameters = {"alpha":[.005, .001], "alpha_l1_1":[1e-5, .0001, .0005, .001], "alpha_l1_2":[1e-7, 1e-6, 5e-6, 1e-5, .0001]}
            #parameters = {"alpha":[.005, .001], "alpha_l1_1":[.0001, .0005, .001], "alpha_l1_2":[1e-5, .0001, .001]}
            

            clf = GridSearchCV(sgd, parameters, scoring='f1', n_jobs=2)
            clf.fit(X_train, y_train)
            #pdb.set_trace()
            print "best alpha = %s; alpha_l1_1 = %s; alpha_l1_2: %s" % (
                clf.best_estimator_.alpha, clf.best_estimator_.alpha_l1_1, clf.best_estimator_.alpha_l1_2)
            preds = clf.predict(X_test)

        if add_interactions and model in ("SGD", "SGDi"):
            zeros, nonzero_i_features = _get_zero_count_i_features(clf, vectorizer, make_table=True)
            
            #if run == 0:
            if return_ifeature_weights:
                for feature_j in nonzero_i_features:
                    # weird reverse lookup..
                    key_index = vectorizer.vocabulary_.values().index(feature_j)
                    f = vectorizer.vocabulary_.keys()[key_index]
                    #if f.startswith("i"):
                    ifeatures[f].append(clf.best_estimator_.coef_[0,feature_j])
                    #ifeatures.append(f)
                            

        fns, fps = [], []
        for i,pred in enumerate(preds):
            if y_test[i] > 0 and pred < 0:
                fns.append(test_sentences[i])

        #print "%s total false negatives." % len(fns)
        with open(fns_out.replace(".txt", "_%s.txt" % run), 'w') as outf:
            outf.write("\n --- \n".join(fns))

        if verbose:
            print db_helper.show_most_informative_features(vectorizer, clf.best_estimator_)

        prec, recall, f, support = sklearn.metrics.precision_recall_fscore_support(
                                        y_test, preds, beta=1)
        accuracy = sklearn.metrics.accuracy_score(y_test, preds)
        
        recalls.append(recall[1])
        precisions.append(prec[1])
        Fs.append(f[1])
        accuracies.append(accuracy)

    if return_ifeature_weights:
        #import operator
        #sorted_features = sorted(ifeatures.iteritems(), reverse=True, key=operator.itemgetter(1))
        #return sorted_features
        return ifeatures

    print "-"*20 + " summary " + "-"*20
    print "model: %s" % model
    #print "penalty: %s" % penalty_str
    print "target: %s out of 3 labeled as ironic" % at_least
    print "add interactions?: %s" % add_interactions
    print "interaction baseline?: %s" % interaction_baseline
    print "add *thread* level interactions?: %s" % add_thread_level_interactions
    print "add sentiment?: %s" % add_sentiment
    print "add *interaction* sentiment? %s" % add_interaction_sentiment

    print "-"*20 + "results" + "-"*20
    print "average F: %s \naverage recall: %s \naverage precision: %s " % (
                avg(Fs), avg(recalls), avg(precisions))
    print "average overall accuracy: %s" % avg(accuracies)
    print "-"*49
    return Fs, recalls, precisions, accuracies


def _make_interaction_vectorizer(all_sentence_ids, sentence_texts, sentence_ids_to_subreddits, 
                                 sentence_ids_to_parses, sentence_ids_to_rows, 
                                 sentence_ids_to_sentiments, add_thread_level_interactions=False,
                                 add_sentiment=True, add_interaction_sentiment=False, sentiment_only=False,
                                 max_features=100000):
    vectorizer = InteractionTermCountVectorizer(
                        ngram_range=(1,2), stop_words="english", binary=False, 
                        max_features=max_features)
    ###
    # figure out sentence indices corresponding to progressive comments
    progressive_indices, conservative_indices = [], []
    for i in xrange(len(all_sentence_ids)):
        s_id = all_sentence_ids[i]

        subreddit_i = sentence_ids_to_subreddits[s_id]
        #pdb.set_trace()
        if subreddit_i == "progressive":
            progressive_indices.append(i)
        else:
            conservative_indices.append(i)

    # turn sentiment on here!
    all_doc_features, NNPs = _make_interaction_features(
            all_sentence_ids, sentence_ids_to_parses,
            sentence_ids_to_rows, sentence_texts, 
            sentence_ids_to_sentiments, 
            add_thread_level_interactions=add_thread_level_interactions,
            sentence_ids_to_subreddits=sentence_ids_to_subreddits,
            add_sentiment=add_sentiment, add_interaction_sentiment=add_interaction_sentiment)

    ### when you only add *simple* interactions
    # and compare to baseline the l1-l2 strategy does
    # well, probably because in this case many features
    # are irrelevant!
  
    # simple 
    # add back progressive + conservative NNPs!
    if not sentiment_only:
        X = vectorizer.fit_transform(sentence_texts, 
                interaction_prefixes=["progressive-NNP", "conservative-NNP"],# "progressive-NNP"],
                interaction_doc_indices=[progressive_indices, conservative_indices],
                interaction_terms = [NNPs, NNPs],
                singleton_doc_features=all_doc_features)
    else:
        print "-- sentiment *only*!"
        X = vectorizer.fit_transform(sentence_texts, 
            interaction_prefixes=[],
            interaction_doc_indices=[],
            interaction_terms = [],
            singleton_doc_features=all_doc_features)

    return X, vectorizer


def _sentence_ids_to_cluster_probs(sentence_ids, 
            users_to_cluster_probs_path="users_to_cluster_probs.pickle"):
    print users_to_cluster_probs_path 
    users_to_cluster_probs = pickle.load(open(users_to_cluster_probs_path))
    sentence_ids_to_users = dict(zip(sentence_ids, [u[0] for u in db_helper.get_user_ids(sentence_ids)]))

    users_to_cluster_probs = pickle.load(open(users_to_cluster_probs_path))
    sentence_ids_to_users = dict(zip(sentence_ids, [u[0] for u in db_helper.get_user_ids(sentence_ids)]))

    sentences_to_cluster_probs = {}
    for id_ in sentence_ids:
        user = sentence_ids_to_users[id_]
        if not user in users_to_cluster_probs:
            print "!!! warning! %s not found!! putting in .5" % user 
            # todo: make sure this is the majority cluster
            #sentence_poster_str = "cluster1" 
            sentences_to_cluster_probs[id_] = .5
        else:
        #    sentence_poster_str = "cluster0" if users_to_clusters[user] == 0 else "cluster1"
            sentences_to_cluster_probs[id_] = users_to_cluster_probs[user][1]

    return sentences_to_cluster_probs
    
def _sentence_ids_to_cluster_strs(sentence_ids, ids_to_clusters_path="sentence_ids_to_speaker_clusters2.pickle",
            users_to_clusters_path="users_to_clusters3.pickle", pickle_it=True):
    
    '''
    if os.path.isfile(ids_to_clusters_path):
        print "found %s -- using it!" % users_to_clusters_path
        #return pickle.load(open(users_to_clusters_path, 'rb'))
        return pickle.load(open(ids_to_clusters_path, 'rb'))
    '''
    #print ids_to_clusters_path
    print users_to_clusters_path

    users_to_clusters = pickle.load(open(users_to_clusters_path))
    sentence_ids_to_users = dict(zip(sentence_ids, [u[0] for u in db_helper.get_user_ids(sentence_ids)]))

    sentences_to_clusters = {}

    for id_ in sentence_ids:
        user = sentence_ids_to_users[id_]
        if not user in users_to_clusters:
            print "!!! warning! %s not found!! putting in default" % user 
            # todo: make sure this is the majority cluster
            sentence_poster_str = "cluster1" 
        else:
            sentence_poster_str = "cluster0" if users_to_clusters[user] == 0 else "cluster1"
        sentences_to_clusters[id_] = sentence_poster_str 

    if pickle_it:
        with open(ids_to_clusters_path, 'wb') as outf:
            pickle.dump(sentences_to_clusters, outf)

    return sentences_to_clusters

def _get_zero_count_i_features(clf, v, make_table=False):
    
    interaction_indices = set(v.get_interaction_term_b(iprefixes=["i-", "f-"]).nonzero()[0])
    wv = clf.best_estimator_.coef_[0]
    nonzero_wvs = set(wv.nonzero()[0])
    nonzero_i_features = nonzero_wvs.intersection(interaction_indices)
    zeros = len(interaction_indices) - len(nonzero_i_features)
    #pdb.set_trace()
    #pdb.set_trace()
    #if make_table:
    #    pdb.set_trace()
    #return zeros, list(nonzero_i_features)
    nonzero_i_features = list(nonzero_i_features)
    return len(nonzero_i_features), nonzero_i_features


def sentence_classification(model="SVC", 
                            add_interactions=False, interaction_baseline=False,
                            cluster_interactions=False, 
                            add_thread_level_interactions=False,
                            verbose=False, tfidf=True, max_features=500000, at_least=2,
                            n_folds=5, seed=30, add_sentiment=False, 
                            add_interaction_sentiment=False,
                            iters=10, 
                            save_feature_weights=False, penalty_str="l2"):
    
    if add_thread_level_interactions and not add_interactions:
        raise Exception, "cannot add thread-level interactions without baseline interactions"

    print "-- sentence classification! ---"


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

    ### 7/9/14
    if cluster_interactions:
        # then swap out sentence_ids_to_reddits with the cluster assignments
        # of the poster
        # 7/15 -- we're going to try and do this 'probabalistically'...
        #all_sentence_ids = sentence_ids_to_subreddits.keys()
        comment_ids_to_cluster_probs = _sentence_ids_to_cluster_probs(all_comment_ids)
        #comment_ids_to_cluster_strs = _sentence_ids_to_cluster_strs(all_comment_ids)
        
        ### 7/16
        #sentence_ids_to_cluster_strs, comments_d, all_sentence_ids = \
        #            _map_sentences_to_cluster_strs(all_comment_ids, comment_ids_to_cluster_probs)
        
        sentence_ids_to_cluster_probs, comments_d, all_sentence_ids = \
            _map_sentences_to_cluster_probs(all_comment_ids, comment_ids_to_cluster_probs)
        #sentence_ids_to_subreddits = sentence_ids_to_cluster_strs

        # overwriting this for now -- everbody gets cluster 1 probs, but 
        #for sentence_id in sentence_ids_to_subreddits:
        #    sentence_ids_to_subreddits[sentence_id] = "probabalistic-cluster1"

    if interaction_baseline: 
        for sentence_id in sentence_ids_to_subreddits:
            # dummy
            sentence_ids_to_subreddits[sentence_id] = "interaction-baseline"

    # this defines how we map from multiple labels (3) to a single
    # summary label
    collapse_f = lambda lbl_set: 1 if lbl_set.count(1) >= at_least else -1
    all_sentence_ids, sentence_texts, sentence_lbls = \
        db_helper.get_texts_and_labels_for_sentences(all_sentence_ids, 
                                        repeat=False, collapse=collapse_f)

    ####
    # set up some convenient dictionaries 
    sentence_ids_to_parses = dict(zip(all_sentence_ids, db_helper.get_parses(all_sentence_ids)))
    sentence_ids_to_sentiments = dict(zip(all_sentence_ids, db_helper.get_sentiments(all_sentence_ids)))
    sentence_ids_to_labels = dict(zip(all_sentence_ids, sentence_lbls))
    sentence_ids_to_rows = dict(zip(all_sentence_ids, range(len(all_sentence_ids))))

    if add_interactions or add_sentiment:
        sentiment_only = False
        if not add_interactions:
            print "sentiment only!!"
            sentiment_only = True

        vectorizer = InteractionTermCountVectorizer(
                            ngram_range=(1,2), stop_words="english", binary=False, 
                            max_features=max_features)
        ###
        # figure out sentence indices corresponding to progressive comments
        progressive_indices, conservative_indices = [], []
        all_indices = range(len(all_sentence_ids))
        for i in all_indices:
            s_id = all_sentence_ids[i]
            subreddit_i = sentence_ids_to_subreddits[s_id]
            ### TODO this is just a hacky way of getting the cluster interactions in...
            #if subreddit_i == "progressive" or cluster_interactions or interaction_baseline:
            if subreddit_i == "progressive" or subreddit_i == "cluster0":
                progressive_indices.append(i)
            else:
                conservative_indices.append(i) # intercept

        # turn sentiment on here!
        if not cluster_interactions:
  
            all_doc_features, NNPs = _make_interaction_features(
                    all_sentence_ids, sentence_ids_to_parses,
                    sentence_ids_to_rows, sentence_texts, 
                    sentence_ids_to_sentiments, 
                    add_thread_level_interactions=add_thread_level_interactions,
                    sentence_ids_to_subreddits=sentence_ids_to_subreddits,
                    add_interaction_sentiment=add_interaction_sentiment,
                    add_sentiment=add_sentiment)

            if not sentiment_only:
                X = vectorizer.fit_transform(sentence_texts, 
                        interaction_prefixes=["progressive-NNP", "Conservative-NNP"],
                        interaction_doc_indices=[progressive_indices, conservative_indices],
                        interaction_terms = [NNPs, NNPs],
                        singleton_doc_features=all_doc_features)
            else:
                print "-- sentiment *only*!"
                X = vectorizer.fit_transform(sentence_texts, 
                        interaction_prefixes=[],
                        interaction_doc_indices=[],
                        interaction_terms = [],
                        singleton_doc_features=all_doc_features)
        else:
            all_doc_features, NNPs = _make_interaction_features(
                    all_sentence_ids, sentence_ids_to_parses,
                    sentence_ids_to_rows, sentence_texts, 
                    sentence_ids_to_sentiments, 
                    add_thread_level_interactions=add_thread_level_interactions,
                    sentence_ids_to_subreddits=sentence_ids_to_subreddits,
                    add_interaction_sentiment=add_interaction_sentiment,
                    add_sentiment=add_sentiment,
                    add_cluster_strs=True)


            X = vectorizer.fit_transform(sentence_texts, 
                    interaction_prefixes=["cluster0-NNP", "cluster1-NNP"],
                    interaction_doc_indices=[all_indices, all_indices],
                    interaction_terms = [NNPs, NNPs],
                    singleton_doc_features=all_doc_features)    
        
            c0_feature_indices = vectorizer.get_indices_for_features_containing(["cluster0-"]).nonzero()[0]
            c1_feature_indices = vectorizer.get_indices_for_features_containing(["cluster1-"]).nonzero()[0]
           
            # probability of each sentence (row) being in cluster1
            #probs_v = np.zeros(len(sentence_ids_to_cluster_probs))
            row_count = 0
            new_vecs = []

            for s_id, row in sentence_ids_to_rows.items():
                if row_count % 1000 == 0:
                    print "on row %s" % row_count
                row_count += 1
                try:
                    p = sentence_ids_to_cluster_probs[s_id]
                except:
                    pdb.set_trace()
                for j in X[row,:].nonzero()[1]:
                    if j in c1_feature_indices:
                        #if p < .5:
                        X[row,j] = X[row,j] * p
                            #X[row,j] = 0.0
                    elif j in c0_feature_indices:
                        #if p > .5:
                        X[row,j] = X[row,j] * (1.0-p)
                        #X[row, j] = 0.0

            print "all done!"

        '''
        # kitchen-sink
        X = vectorizer.fit_transform(sentence_texts, 
                #interaction_prefixes=["conservative-NNP", "progressive-NNP"],
                #interaction_prefixes=["progressive", "conservative"],
                interaction_prefixes=["conservative-NNP", "progressive-NNP", "progressive", "conservative"],
                #interaction_doc_indices=[conservative_indices, progressive_indices],
                #interaction_doc_indices=[progressive_indices, conservative_indices],
                interaction_doc_indices=[conservative_indices, progressive_indices, progressive_indices, conservative_indices],
                #interaction_terms=[NNPs, NNPs],
                #interaction_terms = [None, None],
                interaction_terms=[NNPs, NNPs, None, None],
                singleton_doc_features=all_doc_features)
        '''
    else:
        vectorizer = CountVectorizer(ngram_range=(1,2), stop_words="english", binary=False, 
                                        max_features=max_features)
        X = vectorizer.fit_transform(sentence_texts)


    '''
    # this has been supplanted by the sentiment feature in the interactions...
    if add_sentiment:
        user_to_sentiment, subreddit_to_sentiment = db_helper.get_sentiment_distribution()
        sentence_ids_to_users = db_helper.get_sentence_ids_to_users()
        X0 = scipy.sparse.csr.csr_matrix(np.zeros((X.shape[0], 4)))
        X = scipy.sparse.hstack((X, X0)).tocsr()
        for i in xrange(X.shape[0]):
            sentence_id = all_sentence_ids[i]
            # try:
            #     dist0 = user_to_sentiment[sentence_ids_to_users[sentence_id]]
            # except:
            #     dist0 = np.array([0.2,] * 5)
            try:
                dist1 = subreddit_to_sentiment[sentence_ids_to_subreddits[sentence_id]]
            except:
                dist1 = np.array([0.2,] * 5)
            dist2 = np.array([0.01] * 5)
            dist2[sentence_ids_to_sentiments[sentence_id] + 2] += 0.95
            X[i, X.shape[1] - 1] = 1 if sentence_ids_to_sentiments[sentence_id] <= 0 else -1
            X[i, X.shape[1] - 2] = db_helper.get_sentiment_discrepancy(sentence_id, sentence_ids_to_sentiments)
            X[i, X.shape[1] - 3] = pairwise.cosine_similarity(dist1, dist2)[0][0]
            # the following feature increases recall a lot
            X[i, X.shape[1] - 4] = db_helper.get_upvotes(sentence_id)

            ####### THE BELOW DON"T WORK ######
            #X[i, X.shape[1] - 3] = 1 if sum(dist0[0:2]) > 0.65 and sum(dist2[3:5]) > 0.95 else -1
            #X[i, X.shape[1] - 1] = 1 if pairwise.cosine_similarity(dist0, dist2)[0][0] < 0.65 else -1
            #X[i, X.shape[1] - 3] = 1 if db_helper.kld(dist0, dist2) > 1.68 else -1            
            #X[i, X.shape[1] - 2], X[i, X.shape[1] - 3] = db_helper.get_sentiment_discrepancy(sentence_id, sentence_ids_to_sentiments)

            #tmp = db_helper.length_feature(sentence_id)
            #X[i, X.shape[1] - 1] = 1 if tmp >= 3 and tmp <= 14 else 0
            #X[i, X.shape[1] - 2], X[i, X.shape[1] - 3]= db_helper.get_sentiment_discrepancy(sentence_id, sentence_ids_to_sentiments)
    '''

    # row normalize features. Make sure that you are not normalizing twice. 
    # Commnet out lines 993 and 994 in sklearn/feature_extract/text.py. 
    # Or add features and run tfidf afterwards and get rid of this if-clause.
    # if tfidf:
    #     X = normalize(X, norm='l2', copy=False)

    if tfidf:
        transformer = TfidfTransformer()
        X = transformer.fit_transform(X)


    '''
    if cluster_interactions:
        c1_feature_indices = vectorizer.get_indices_for_features_containing(["cluster1-"]).nonzero()[0]
        
        # probability of each sentence (row) being in cluster1
        probs_v = np.zeros(len(sentence_ids_to_cluster_probs))
        row_count = 0
        new_vecs = []
        for s_id, row in sentence_ids_to_rows.items():
            if row_count % 100 == 0:
                print "on row %s" % row_count
            row_count += 1
            p = sentence_ids_to_cluster_probs[s_id]
            #X[row,c1_feature_indices] = X[row,c1_feature_indices] * p 
            #probs_v[row] = sentence_ids_to_cluster_probs[s_id]
            #cur_X = X[row,:]
            #pdb.set_trace()
            for j in X[row,:].nonzero()[1]:
                if j in c1_feature_indices:
                    #cur_X[j] = cur_X[j] * p
                    X[row,j] = X[row,j] * p
            #new_vecs.append(X[row,c1_feature_indices] * p)
            #new_vecs.append(cur_X)
        print "all done!"
    '''

    ####
    # ok -- now we cross-fold validate
    #
    # bcw -- 4/15 -- moving to 'bootstrap'
    ####
    #kf = KFold(len(all_comment_ids), n_folds=n_folds, shuffle=True, random_state=seed)
    
    feature_weights = defaultdict(list) ## save feature weights across splits.
    # record all metrics for each train/test split
    # (this is what we use for our empirical counts)
    recalls, precisions, Fs, AUCs = [], [], [], []
    accuracies = []
    zero_counts = []
    #for train, test in kf:
    cur_iter = 0
    
    N_comments = len(all_comment_ids)
    while cur_iter < iters:
        if (cur_iter+1) % 100 == 0:
            print "on iter %s" % (cur_iter + 1)
        # we fix the seed so that results are comparable! 
        train, test = sklearn.cross_validation.train_test_split(range(N_comments), test_size=.2, 
                        random_state=seed * (cur_iter+1))
        train_comment_ids = db_helper._get_entries(all_comment_ids, train)
        train_rows, train_sentences, y_train = _get_rows_and_y_for_comments(train_comment_ids, 
                                comments_d, sentence_ids_to_rows, sentence_ids_to_labels)

        test_comment_ids = db_helper._get_entries(all_comment_ids, test)
        test_rows, test_sentences, y_test = _get_rows_and_y_for_comments(test_comment_ids, 
                                comments_d, sentence_ids_to_rows, sentence_ids_to_labels)
        X_train, X_test = X[train_rows], X[test_rows]
        
        ### TMP TMP TMP
        #test_rows1, test_sentences1, y_test1 = _get_rows_and_y_for_comments(test_comment_ids, 
        #                        comments_d, sentence_ids_to_rows, sentence_ids_to_labels1)

        dec_values = None
        if model == "SVC":
            svc = LinearSVC(loss="l2", penalty="l2", dual=False, class_weight="auto")
            parameters = {'C':[ .00001, .0001, .001, .01]}
            clf = GridSearchCV(svc, parameters, scoring='f1')
            clf.fit(X_train, y_train)
            dec_values = clf.decision_function(X_test)

            preds = sgn(dec_values)
        elif model == "baseline":
            # guess at chance
            p_train = len([y_i for y_i in y_train if y_i > 0])/float(len(y_train))
            def baseline_clf(): # no input!
                if random.random() < p_train:
                    return 1
                return -1
            preds = [baseline_clf() for i in xrange(len(y_test))]
        elif model == "SGD":
            sgd = stochastic_gradient.SGDClassifier(loss="log", class_weight="auto", penalty=penalty_str)
            parameters = {"alpha":[.005, .001]}
            clf = GridSearchCV(sgd, parameters, scoring='f1')
            #sgd.fit(X_train, y_train)

            clf.fit(X_train, y_train)
            #print clf.best_params_
        
            
            # interaction

        

            preds = clf.predict(X_test)
            #import pdb; pdb.set_trace()
        elif model == "SGDi":
            penalty_str = "l1l2"
            #b = vectorizer.get_interaction_term_b(iprefixes=["i-", "f-"], 
            #                                        exclude_containing=["overall-"])
            b = vectorizer.get_interaction_term_b(iprefixes=["i-"])
            b2 = vectorizer.get_interaction_term_b(iprefixes=["f-"], exclude_containing=["overall-"])
            #pdb.set_trace()
            #sgd = stochastic_gradient_i.SGDClassifier(l1_indicator_v=b,
            #    loss="log", class_weight="auto", penalty=penalty_str)

            #parameters = {"alpha":[.005, .0001, .001], 
            #              "alpha_l1":[5e-5, .0001,  .00025, .0005]}
            #"alpha_l1":[5e-5, .0001,  .00025, .0005]

            sgd = stochastic_gradient_i2.SGDClassifier(l1_indicator_v1=b, l1_indicator_v2=b2,
                                                        loss="log", penalty=penalty_str, 
                                                        class_weight="auto")       
                # [5e-5, 1e-5,  1e-4, 1e-3] 
            parameters = {"alpha":[.005, .001], "alpha_l1_1":[1e-5, .0001, .0005, .001], "alpha_l1_2":[1e-7, 1e-6, 5e-6, 1e-5, .0001]}
            

            clf = GridSearchCV(sgd, parameters, scoring='f1', n_jobs=2)
            clf.fit(X_train, y_train)
            #pdb.set_trace()
            print "best alpha = %s; alpha_l1_1 = %s; alpha_l1_2: %s" % (
                clf.best_estimator_.alpha, clf.best_estimator_.alpha_l1_1, clf.best_estimator_.alpha_l1_2)
            preds = clf.predict(X_test)

        if verbose:
            print db_helper.show_most_informative_features(vectorizer, clf.best_estimator_)

        if save_feature_weights:
            ranked_features = db_helper.show_most_informative_features(
                                vectorizer, clf.best_estimator_, return_sorted_list=True)
            
            for w, feature in ranked_features:
                feature_weights[feature].append(w)


        if verbose:
            print sklearn.metrics.classification_report(y_test, preds)

        prec, recall, f, support = sklearn.metrics.precision_recall_fscore_support(
                                    y_test, preds, beta=1)
        recalls.append(recall[1])
        
        if add_interactions and model in ("SGD", "SGDi"):
            zeros, nonzero_i_features = _get_zero_count_i_features(clf, vectorizer)
            #print zeros
            zero_counts.append(zeros)
            ifeatures = []

            '''
            for feature_j in nonzero_i_features:
                # weird reverse lookup..
                key_index = vectorizer.vocabulary_.values().index(feature_j)
                ifeatures.append(vectorizer.vocabulary_.keys()[key_index])
            '''
            #pdb.set_trace()
        precisions.append(prec[1])
        Fs.append(f[1])
        accuracy = sklearn.metrics.accuracy_score(y_test, preds)
        accuracies.append(accuracy)
        if dec_values is not None:
            precision, recall, thresholds = precision_recall_curve(y_test, dec_values)
            area = auc(recall, precision)
            AUCs.append(area)
        else:
            AUCs.append(0.0)
        cur_iter += 1


    print "-"*20 + " summary " + "-"*20
    print "model: %s" % model
    print "penalty: %s" % penalty_str
    print "target: %s out of 3 labeled as ironic" % at_least
    print "add interactions?: %s" % add_interactions
    print "cluster interactions?: %s" % cluster_interactions
    print "baseline interactions?: %s" % interaction_baseline
    print "add *thread* level interactions?: %s" % add_thread_level_interactions
    print "add sentiment?: %s" % add_sentiment
    print "add *interaction* sentiment?: %s" % add_interaction_sentiment
    print "-"*20 + "results" + "-"*20
    print "average F: %s \naverage recall: %s \naverage precision: %s " % (
                avg(Fs), avg(recalls), avg(precisions))
    print "average AUC: %s" % avg(AUCs)
    if len(zero_counts)>0:
        print "average zero count: %s" % avg(zero_counts)
    print "average overall accuracy: %s" % avg(accuracies)
    print "-"*49
    return Fs, recalls, precisions, feature_weights, zero_counts


def _get_rows_and_y_for_comments(comment_ids, comments_d, sentence_ids_to_rows, 
                                    sentence_ids_to_labels):
    rows, y, y2 = [], [], []
    sentences = []
    for comment in comment_ids:
        sentence_ids = comments_d[comment]["sentence_ids"]
        for sent_id in sentence_ids:
            rows.append(sentence_ids_to_rows[sent_id])
            y.append(sentence_ids_to_labels[sent_id])
            #y2.append(sentence_ids_to_labels2[sent_id])

        sentences.extend(db_helper.grab_segments(sentence_ids))
    
    return rows,sentences,y#,y2

def _make_interaction_features(all_sentence_ids, sentence_ids_to_parses, 
                                sentence_ids_to_rows, sentence_texts, 
                                sentence_ids_to_sentiments, 
                                add_interaction_sentiment=False,
                                add_thread_level_interactions=False, 
                                sentence_ids_to_subreddits=None,
                                add_sentiment=True, add_cluster_strs=False):
        
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

        ### 
        # replaced DK's engineering with this!
        if add_sentiment:
            if sentence_ids_to_sentiments[sent_id] > 0:
                cur_doc_features.append("overall-sentiment-positive")

            if sentence_ids_to_sentiments[sent_id] > 1:
                cur_doc_features.append("overall-sentence-REALLY-positive")

        #print "--- sent id: %s ---" % sent_id
        for sent_NNP in sent_NNPs:
            s = sent_NNP.lower()

            #    pdb.set_trace()
            index_ = sentence_ids_to_rows[sent_id]
            sent_text = sentence_texts[index_]

            '''
            if s == "way" and sent_NNP in sent_text:
                # 13151
                print "--- ----"
                print "way in %s!\n" % sent_text
                thread_title, thread_id = db_helper.get_thread_title_and_id(sent_id) 
                print "way in %s (%s)" % (sent_id, sentence_ids_to_subreddits[sent_id])
                print "thread id: %s; title: %s" % (thread_id, thread_title)
                print sent_text
                print "--- ----"
                #pdb.set_trace() 
            '''

            '''
            if s == "bravo":
                print "bravo in %s!\n" % sent_text
                thread_title, thread_id = db_helper.get_thread_title_and_id(sent_id) 
                print "bravo in %s (%s)" % (sent_id, sentence_ids_to_subreddits[sent_id])
                print "thread id: %s; title: %s" % (thread_id, thread_title)
                print sent_text
                print "\n" 
            '''

            # note that the first check here is redundant, because
            # if we're not including thread features, we would not
            # see NNP's that are not in the comment text. but 
            # better to be explicit
            if add_thread_level_interactions and sent_NNP not in sent_text:
                # then this NNP was extracted from the reddit *thread*
                # not the actual *comment* -- here we add interaction features
                # that cross NNP's in threads with sentiment

                NNPs.append("overall-comment-%s" % s) # thread level 

                if add_interaction_sentiment and sentence_ids_to_sentiments[sent_id] > 0:
                    # 7/16/14
                    if add_cluster_strs:
                        cur_doc_features.append("comment-cluster0-%s-positive" % s)
                        cur_doc_features.append("comment-cluster1-%s-positive" % s)

                    else:
                        cur_doc_features.append("comment-%s-%s-positive" % 
                                   (sentence_ids_to_subreddits[sent_id], s))    
                    
                    thread_title, thread_id = db_helper.get_thread_title_and_id(sent_id) 
                    '''
                    if s=="way":
                        pdb.set_trace()
                        print "way in %s (%s)" % (sent_id, sentence_ids_to_subreddits[sent_id])
                        print "thread id: %s; title: %s" % (thread_id, thread_title)
                        print sent_text
                        print "\n"
                    

                    if s=="god":
                        print "god in %s (%s)" % (sent_id, sentence_ids_to_subreddits[sent_id])
                        print "thread id: %s; title: %s" % (thread_id, thread_title)
                        print sent_text
                        print "\n"
                  

                    if s=="cruz":
                        print "cruz in %s (%s)" % (sent_id, sentence_ids_to_subreddits[sent_id])
                        print "thread id: %s title: %s" % (thread_id, thread_title)
                        print sent_text
                        print "\n"
                        #pdb.set_trace()
                    
                   
                    if s=="palin":
                        print "palin in %s (%s)" % (sent_id, sentence_ids_to_subreddits[sent_id])
                        print "thread id: %s title: %s" % (thread_id, thread_title)
                        print sent_text
                        print "\n"
                    '''

            elif sent_NNP in sent_text:
                NNPs.append(s)

            
            if add_interaction_sentiment and sent_NNP in sent_text:
                #if s=="cruz":
                #    print "cruz in %s (%s)" % (sent_id, sentence_ids_to_subreddits[sent_id])
                    #print "thread id: %s title: %s" % (thread_id, thread_title)
                #    print sent_text
                #    print "\n"
                #    pdb.set_trace()
                '''
                if sentence_ids_to_subreddits[sent_id] == "Conservative" and \
                        sentence_ids_to_sentiments[sent_id] > 0 and s in (
                        "cruz", "mr", "russia", "king", "oprah", "science", "math", "america"):
                    print "\n\n --- \n"
                    print s
                    print sent_text
                    print " --- "

                elif sentence_ids_to_subreddits[sent_id] == "progressive" and \
                        sentence_ids_to_sentiments[sent_id] > 0 and s in (
                        "war", "times*", "obama", "american", "magic", "where", "world"):
                    print "\n\n --- PROGRESSIVE --- \n"
                    print s
                    print sent_text
                    print " --- END PROGRESSIVE --- "
                '''
                if sentence_ids_to_sentiments[sent_id] > 0:
        
                    #pass#cur_doc_features
                    if add_cluster_strs: # 7/16
                        cur_doc_features.append("sentence-cluster0-%s-positive" % s)
                        cur_doc_features.append("sentence-cluster1-%s-positive" % s)
                    else:
                        cur_doc_features.append("sentence-%s-%s-positive" % 
                                   (sentence_ids_to_subreddits[sent_id], s)) 
           
        all_doc_features_d[sent_id] = cur_doc_features

    NNPs = list(set(NNPs))
    all_doc_features = []
    for id_ in all_sentence_ids:
        all_doc_features.append(all_doc_features_d[id_])

    return all_doc_features, NNPs

def _map_sentences_to_cluster_probs(all_comment_ids, comments_to_cluster_probs):
    sent_ids_to_cluster_probs = {}
    comments_d = {} # point from comment_id to sentences, etc.
    all_sentence_ids = [] # keep a flat list of sentence ids, too
    for comment_id in all_comment_ids:
        comment_sentence_ids, comment_subreddit = db_helper.get_sentence_ids_for_comment(comment_id)
        comments_d[comment_id] = {"sentence_ids":comment_sentence_ids, 
                                    "subreddit":comment_subreddit,
                                    "cluster":comments_to_cluster_probs[comment_id]}
        for sent_id in comment_sentence_ids:
            sent_ids_to_cluster_probs[sent_id] = comments_to_cluster_probs[comment_id]
            all_sentence_ids.append(sent_id)
    return sent_ids_to_cluster_probs, comments_d, all_sentence_ids   

def _map_sentences_to_cluster_strs(all_comment_ids, comments_to_cluster_strs):
    sent_ids_to_cluster_strs = {}
    comments_d = {} # point from comment_id to sentences, etc.
    all_sentence_ids = [] # keep a flat list of sentence ids, too
    for comment_id in all_comment_ids:
        comment_sentence_ids, comment_subreddit = db_helper.get_sentence_ids_for_comment(comment_id)
        comments_d[comment_id] = {"sentence_ids":comment_sentence_ids, 
                                    "subreddit":comment_subreddit,
                                    "cluster":comments_to_cluster_strs[comment_id]}
        for sent_id in comment_sentence_ids:
            sent_ids_to_cluster_strs[sent_id] = comments_to_cluster_strs[comment_id]
            all_sentence_ids.append(sent_id)
    return sent_ids_to_cluster_strs, comments_d, all_sentence_ids   

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

def _feature_dictionary_to_means(features):
    feature_means = {}
    for f, vals in features.items():
        feature_means[f] = np.median(vals)#np.mean(vals)

    return feature_means

def get_top_features(feature_weights_dict, n=100):
    f_dict = _feature_dictionary_to_means(feature_weights_dict)
    #feature_absolute_vals = {}
    #for f, mean in f_dict.items():
    #    feature_absolute_vals[f] = abs(mean)
    
    sorted_x = sorted(feature_weights_dict.iteritems(), 
                        key=operator.itemgetter(1), reverse=True)
    # most positive features, most negative features
    return [x[0] for x in sorted_x[:n]], [x[0] for x in sorted_x[-n:]]

def run_irony_experiments(iters=500, at_least=2):
    
    print "baseline"
    F_baseline_svm, recalls_baseline_svm, precisions_baseline_svm, features_baseline_svm = \
            sentence_classification(iters=iters, save_feature_weights=True, verbose=False, 
                                    at_least=at_least)

    
    
    print "\n" + "-"*50 + "\n"
    print "interactions, no sentiment"
    F_interactions, recalls_interactions, precisions_interactions, features_interactions = \
            sentence_classification(add_interactions=True, add_thread_level_interactions=True, 
                add_sentiment=False, iters=iters, save_feature_weights=True, verbose=False,
                at_least=at_least)
  
    print "\n" + "-"*50 + "\n"
    print "sentiment, no interactions"
    F_interactions_sent, recalls_interactions_sent, precisions_interactions_sent, features_interactions_sent = \
            sentence_classification(add_interactions=False, add_thread_level_interactions=False, 
                add_sentiment=True, iters=iters, save_feature_weights=True, at_least=at_least)

    print "\n" + "-"*50 + "\n"
    print "interactions + sentiment"
    F_interactions_sent, recalls_interactions_sent, precisions_interactions_sent, features_interactions_sent = \
            sentence_classification(add_interactions=True, add_thread_level_interactions=True, 
                add_sentiment=True, iters=iters, save_feature_weights=True, at_least=at_least)



