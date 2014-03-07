import pdb
import sqlite3
import sys
import collections
from collections import defaultdict
import re
import itertools

import nltk # for metrics

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="nogrid")
sns.color_palette("deep")

import numpy as np
import statsmodels.api as sm

import search_reddit

'''
general @TODO you need to decide how to deal with comments 
labeled by different annotators! right now you are limiting
(in most places) to those labeled by the same people --
i.e., this is you use in the 'agreement' function
'''

#db_path = "/Users/bwallace/dev/computational-irony/data-11-30/ironate.db"
####
# change me!
####
db_path = "/Users/bwallace/dev/computational-irony/data-2-7/ironate.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
### we consider 4 active labelers
labelers_of_interest = [2,4,5,6] # TMP TMP TMP

comment_sep_str = "\n\n------------------------------------------\n"

def _make_sql_list_str(ls):
    return "(" + ",".join([str(x_i) for x_i in ls]) + ")"

labeler_id_str = _make_sql_list_str(labelers_of_interest)

def _grab_single_element(result_set, COL=0):
    return [x[COL] for x in result_set]

def _all_same(items):
    return all(x == items[0] for x in items)

def disagreed_to_disk(fout="disagreements.txt"):
    disagreement_ids, comments_to_lbls = disagreed_upon()
    conflicted_comments = grab_comments(disagreement_ids)
    out_str = []
    for i,id_ in enumerate(disagreement_ids):
        out_str.append("{0}comment id: {1}\nlabels: {2}\n{3}".format(
            comment_sep_str, id_, comments_to_lbls[id_], conflicted_comments[i]))
    
    with open(fout, 'w') as out_file:
        out_file.write("\n".join(out_str))

def disagreed_upon():
    task, tuples, comments_to_lbls = agreement()
    disagreement_ids = []
    for id_, lbls in comments_to_lbls.items():
        if not _all_same(lbls):
            disagreement_ids.append(id_)
    return disagreement_ids, comments_to_lbls

def uniform_irony_to_disk(fout="uniformly_ironic.txt"):
    irony_ids, comments_to_lbls = uniform_irony()
    uniformly_ironic_comments = grab_comments(irony_ids)
    out_str = []
    for i,id_ in enumerate(irony_ids):
        out_str.append("{0}comment id: {1}\nlabels: {2}\n{3}".format(
            comment_sep_str, id_, comments_to_lbls[id_], uniformly_ironic_comments[i]))
    
    with open(fout, 'w') as out_file:
        out_file.write("\n".join(out_str))

def subreddit_breakdown():
    pass


def uniform_irony():
    task, tuples, comments_to_lbls = agreement()
    uniformly_ironic_ids = []
    for id_, lbls in comments_to_lbls.items():
        if _all_same(lbls) and lbls[0]==1:
            uniformly_ironic_ids.append(id_)
    return uniformly_ironic_ids, comments_to_lbls

def any_irony():
    task, tuples, comments_to_lbls = agreement()
    at_least_one_vote = []
    for id_, lbls in comments_to_lbls.items():
        if any([y_i==1 for y_i in lbls]):
            at_least_one_vote.append(id_)
    return at_least_one_vote, comments_to_lbls


def get_labeled_thrice_comments():
    cursor.execute(
        '''select comment_id from irony_label group by comment_id having count(labeler_id) >= 3;'''
    )
    thricely_labeled_comment_ids = _grab_single_element(cursor.fetchall())
    return thricely_labeled_comment_ids

def descriptive_stats():
    thricely_labeled_comment_ids = get_labeled_thrice_comments()
    thricely_ids_str = _make_sql_list_str(thricely_labeled_comment_ids)

    subreddits =  _grab_single_element(cursor.execute(
                '''select subreddit from irony_comment where 
                        id in %s;''' % thricely_ids_str)) 

    segment_counts = _grab_single_element(cursor.execute(
            ''' select count(DISTINCT segment_id) from (select * from irony_label where
                    comment_id in %s);''' % thricely_ids_str))

                #'''select subreddit from irony_comment where 
                #        id in %s;''' % thricely_ids_str))
                #)


def pairwise_kappa():
    pairwise_kappas = []
    for labeler_set in itertools.permutations(labelers_of_interest, 2):
        try:
            task, tuples, comments_to_summary_lbls, comments_to_labeler_sets = \
                agreement(these_labelers=set([str(l) for l in labeler_set]))
        except:
            pdb.set_trace()
        pairwise_kappas.append(task.kappa())
    return sum(pairwise_kappas)/float(len(pairwise_kappas))

# e.g., pairwise annotation task between "4" and "5" like so:
# task, tuples, comments_to_lbls, comments_to_lblers = annotation_stats.agreement(these_labelers=set(["4","5"])
def agreement(these_labelers=None):
    # select all segment labels
    cursor.execute(
        '''select comment_id, segment_id, labeler_id, label from irony_label 
            where forced_decision=0 and labeler_id in %s;''' % labeler_id_str)
    all_segments = cursor.fetchall()

    comments_to_labels = defaultdict(list)
    comments_to_labeler_ids = defaultdict(list)
    for seg in all_segments:
        comment_id, segment_id, labeler, label = seg
        labeler = str(labeler)
        comments_to_labels[comment_id].append((labeler, str(segment_id), str(label)))
        comments_to_labeler_ids[comment_id].append(labeler)

    comments_to_labeler_sets = {}
    for comment_id, labeler_list in comments_to_labeler_ids.items():
        comments_to_labeler_sets[comment_id] = set(labeler_list)

    if these_labelers is None:
        these_labelers = set([str(l) for l in labelers_of_interest])

    # nltk wants tuples like: ('c1', '1', 'v1')
    tuples = []
    comments_to_summary_lbls = defaultdict(list)
    # now collapse 
    for i, comment_id in enumerate(comments_to_labels):
        labelers = comments_to_labeler_sets[comment_id]
        # @TODO not sure if this is how you want to deal with this, in general
        #if labelers == set([str(l) for l in labelers_of_interest]):
        if these_labelers.issubset(labelers):
            comment_labels = comments_to_labels[comment_id]
            was_labeled_by = lambda a_label, a_labeler : a_label[0] == a_labeler
            for labeler in these_labelers:
                labeler_labels = [
                    int(lbl_tuple[-1]) for lbl_tuple in comment_labels if
                         was_labeled_by(lbl_tuple, labeler)]
                # call it a 1 if any were 1 (for this labeler!) 
                labeler_aggregate_label = max(labeler_labels)
                comments_to_summary_lbls[str(comment_id)].append(labeler_aggregate_label)
                cur_tuple = (str(labeler), str(comment_id), str(labeler_aggregate_label))
                tuples.append(cur_tuple)

    task = nltk.AnnotationTask(data=tuples)

    print "kappa is: {0}".format(task.kappa())
    #pdb.set_trace()
    return task, tuples, comments_to_summary_lbls, comments_to_labeler_sets


def plot_confidence_changes():
    from matplotlib import rc
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)

    confidence_dict = confidence_change_forced()
    ### average over labelers
    labeler_changed_mind_d = defaultdict(list)
    labeler_conf_delta = defaultdict(list)
    
    labeler_delta_plot_d = defaultdict(dict)
    for labeler in labelers_of_interest:
        labeler_delta_plot_d[labeler] = {
                            "00":{"count":0, "confidences":[]},
                            "11":{"count":0, "confidences":[]},
                            "01":{"count":0, "confidences":[]},
                            "10":{"count":0, "confidences":[]}}

    bool_to_dummy = lambda x: 1 if x>0 else 0
    decisions_to_str = lambda d0, d1: "%s%s" % (
                            bool_to_dummy(d0), bool_to_dummy(d1))


    before_after_counts = {"00":0, "11":0, "01":0, "10":0}
    better_titles = {"00":r"unironic $\rightarrow$ unironic", 
                     "11":"blah", "01":"blah", "10":"10"}
                        
                        #"final":(final_max_lbl, final_conf),
                        #"forced": (forced_max_lbl, forced_conf) 

    for comment in confidence_dict:
        for labeler, lbl_dict in confidence_dict[comment].items():
            d0 = lbl_dict['forced'][0]
            d1 = lbl_dict['final'][0]
            changed_mind = d0!=d1
            labeler_changed_mind_d[labeler].append(bool_to_dummy(changed_mind))

            c0 = lbl_dict['forced'][1]
            c1 = lbl_dict['final'][1] 
            conf_delta = c1 - c0
            labeler_conf_delta[labeler].append(conf_delta)
            
            box_str = decisions_to_str(d0, d1)
            #pdb.set_trace()
            labeler_delta_plot_d[labeler][box_str]["count"] += 1
            labeler_delta_plot_d[labeler][box_str]["confidences"].append((c0, c1))

    for labeler in labelers_of_interest:
        plt.clf()
        decision_strs = before_after_counts.keys()
        labeler_data = labeler_delta_plot_d[labeler]
        
        z_conf = 3.0
        #for decision_str in decision_strs:
        #    z_conf = max()
        from matplotlib import gridspec
        # gs =  gridspec.GridSpec(4, 1, height_ratios=[1, 1 ,1.5, 1])
        frame, axes = plt.subplots(4, 2, sharex=True, sharey=True)
        frame.set_figwidth(5)
        #frame.set_figwidth(4)

        frame.subplots_adjust(hspace=0, wspace=0)
        plt.setp([a.get_xticklabels() for a in axes[0, :]], visible=False)

        # bottom two
        for j in xrange(2):
            axes[-1][j].xaxis.set_ticks([])

        axes[-1][0].set_xlabel("forced decision", size=22)
        axes[-1][1].set_xlabel("final decision", size=22)

        #axes[-1][1].xaxis.set_visible(False)
        cool, warm = sns.color_palette("coolwarm", 2) 
        Blues = plt.get_cmap('Blues')
        Reds = plt.get_cmap('Reds')
        Grays = plt.get_cmap('gray')
        for i in xrange(4):
            decision_data = labeler_data[decision_strs[i]]
            count = decision_data['count']
            axes[i][0].yaxis.set_ticks([])#set_visible(False)
            #axes[i][0].set_ylabel("%s \n (%s)" % 
            #            (better_titles[decision_strs[i]], count), 
            #            rotation="horizontal")
            axes[i][0].set_ylabel("%s" % count, rotation="horizontal", size=18)
            axes[i][0].yaxis.set_label_position("right")

            confidences = []
            # before
            conf0 = [c[0] for c in decision_data["confidences"]]
            avg_conf0 = sum(conf0)/float(len(conf0))

            if i % 2 == 0:
                color0 = Blues(avg_conf0)
                #color0 = Blacks()
            else:
                color0 = Reds(avg_conf0)
            #color0 = Grays(avg_conf0)
            axes[i][0].set_axis_bgcolor(color0)

            # after
            conf1 = [c[1] for c in decision_data["confidences"]]
            avg_conf1 = sum(conf1)/float(len(conf1))

            if i % 2 == 0:
                color1 = Blues(avg_conf1)
            else:
                color1 = Reds(avg_conf1)
            #color1 = Grays(avg_conf1)
            axes[i][1].set_axis_bgcolor(color1)
            #
        #frame.tight_layout()
        plt.savefig("%s.pdf" % labeler)
        print decision_strs

def confidence_change_forced():

    triply_labeled_comment_ids = get_labeled_thrice_comments()
    triply_labeled_comments_str = _make_sql_list_str(triply_labeled_comment_ids)

    final_decisions =  list(cursor.execute(
        '''select labeler_id, comment_id, label, confidence from irony_label 
            where forced_decision=0 and comment_id in %s and labeler_id in %s;''' % 
            (triply_labeled_comments_str, labeler_id_str)))



    # collapse final decision into dictionary mapping comment ids to 
    # dictionaries that in turn map to decisions
    final_labels_dict = defaultdict(lambda: defaultdict(list))
    comments = []
    for labeler_id, comment_id, label, confidence in final_decisions:
        final_labels_dict[comment_id][labeler_id].append((label, confidence))
        comments.append(comment_id)

    comments = list(set(comments))

    ###
    # forced labels!
    forced_decisions = list(cursor.execute(
                '''select labeler_id, comment_id, label, confidence from irony_label where 
                    forced_decision=1 and comment_id in %s and labeler_id in %s;''' % 
                    (triply_labeled_comments_str, labeler_id_str)))

    forced_labels_dict = defaultdict(lambda: defaultdict(list))
    for labeler_id, comment_id, label, confidence in forced_decisions:
        forced_labels_dict[comment_id][labeler_id].append((label, confidence))
        


    # now just call the comment a 1 if any segment in it
    # is a 1
    collapsed_final_labels_dict = defaultdict(lambda: defaultdict(list))
    collapsed_forced_labels_dict = defaultdict(lambda: defaultdict(list))
    final_and_forced_dict = defaultdict(lambda: defaultdict(dict))
    filter_these = []
    for comment in comments:
        if comment in forced_labels_dict.keys():
            for labeler in labelers_of_interest:
                if len(forced_labels_dict[comment][labeler]) > 0 and len(
                        final_labels_dict[comment][labeler]):
                    final_max_lbl = max([x[0] for x in final_labels_dict[comment][labeler]])
                    final_conf = final_labels_dict[comment][labeler][0][1]
                    forced_max_lbl = max([x[0] for x in forced_labels_dict[comment][labeler]])
                    forced_conf = forced_labels_dict[comment][labeler][0][1]
                    final_and_forced_dict[comment][labeler] = {
                        "final":(final_max_lbl, final_conf),
                        "forced": (forced_max_lbl, forced_conf) 
                    }

                '''
                if len(final_labels_dict[comment][labeler]) > 0:
                    final_max_lbl = max([x[0] for x in final_labels_dict[comment][labeler]])
                    # these will be the same for all segments!
                    final_conf = final_labels_dict[comment][labeler][0][1]
                    collapsed_final_labels_dict[comment][labeler] = (max_lbl, conf)

                if len(forced_labels_dict[comment][labeler]) > 0:
                    forced_max_lbl = max([x[0] for x in forced_labels_dict[comment][labeler]])
                    forced_conf = forced_labels_dict[comment][labeler][0][1]
                    collapsed_forced_labels_dict[comment][labeler] = (max_lbl, conf)
                '''

            #else:
            #    pdb.set_trace()
            #    filter_these.append(comment)


    
    #return collapsed_labels_dict, ironic_comments
    # now, look at
    #return collapsed_forced_labels_dict, collapsed_final_labels_dict
    return final_and_forced_dict

def n_users_requested_context_for_comment(comment_id):
    ''' how many annotators requested context for this comment? '''
    forced_decisions = _grab_single_element(cursor.execute(
                '''select distinct labeler_id from irony_label where forced_decision=1 and comment_id =%s and labeler_id in %s;''' % 
                   (comment_id, labeler_id_str)))
    return forced_decisions

def n_users_labeled_as_irony(comment_id):
    ''' how many annotators labeled this comment as containing some irony? '''
    ironic_lblers = _grab_single_element(cursor.execute(
        '''select distinct labeler_id from irony_label 
            where forced_decision=0 and comment_id = %s and label=1 and labeler_id in %s;''' % 
            (comment_id, labeler_id_str)))
    return ironic_lblers

def get_forced_comments():
    # pre-context / forced decisions
    forced_decisions = _grab_single_element(cursor.execute(
                '''select distinct comment_id from irony_label where forced_decision=1 and labeler_id in %s;''' % 
                    labeler_id_str)) 

    forced_decisions_str = _make_sql_list_str(forced_decisions)
    comment_urls = cursor.execute(
                '''select id, thread_url, permalink, subreddit from irony_comment where id in %s;''' % 
                forced_decisions_str).fetchall()


    #pdb.set_trace()
    #p_forced = float(len(forced_decisions)) / float(len(all_comment_ids))

    # now look at the proportion forced for the ironic comments
    ironic_comments = get_ironic_comment_ids()

    # which were labeled ironic?
    irony_labels = []
    for comment_id in forced_decisions:
        if comment_id in ironic_comments:
            irony_labels.append(1)
        else:
            irony_labels.append(0)

    ironic_ids_str = _make_sql_list_str(ironic_comments)
    forced_ironic_ids =  _grab_single_element(cursor.execute(
                '''select distinct comment_id from irony_label where 
                        forced_decision=1 and comment_id in %s and labeler_id in %s;''' % 
                                (ironic_ids_str, labeler_id_str))) 
    #return dict(zip(forced_ironic_ids, grab_comments(forced_ironic_ids)))
    return zip(grab_comments(forced_decisions), irony_labels, comment_urls)

def get_ironic_comment_ids():
    cursor.execute(
        '''select distinct comment_id from irony_label 
            where forced_decision=0 and label=1 and labeler_id in %s;''' % 
            labeler_id_str)

    ironic_comments = _grab_single_element(cursor.fetchall())
    return ironic_comments

def naive_irony_prop():
    # comments that at least one person has labeled one sentence within as being 
    # ironic
    #cursor.execute("select distinct comment_id from irony_label where forced_decision=0 and label=1;")
    # restricting to a subset of lablers for now!
    ironic_comments = get_ironic_comment_ids()

    # all comments
    cursor.execute(
        '''select distinct comment_id from irony_label where forced_decision=0 and
            labeler_id in %s;''' % labeler_id_str)
    all_comments = _grab_single_element(cursor.fetchall())

    n_ironic = float(len(ironic_comments))
    N = float(len(all_comments))
    return n_ironic, N, ironic_comments, all_comments

def get_all_comment_ids():
    return _grab_single_element(cursor.execute(
                '''select distinct comment_id from irony_label where labeler_id in %s;''' % 
                    labeler_id_str)) 

def context_stats():
    all_comment_ids = get_all_comment_ids()

    # pre-context / forced decisions
    forced_decisions = _grab_single_element(cursor.execute(
                '''select distinct comment_id from irony_label where forced_decision=1 and labeler_id in %s;''' % 
                    labeler_id_str)) 

    for labeler in labelers_of_interest:
        labeler_forced_decisions = _grab_single_element(cursor.execute(
                '''select distinct comment_id from irony_label where forced_decision=1 and labeler_id = %s;''' % 
                    labeler))

        all_labeler_decisions = _grab_single_element(cursor.execute(
                '''select distinct comment_id from irony_label where forced_decision=0 and labeler_id = %s;''' % 
                    labeler))

        p_labeler_forced = float(len(labeler_forced_decisions))/float(len(all_labeler_decisions))
        print "labeler %s: %s" % (labeler, p_labeler_forced)

    pdb.set_trace()
    p_forced = float(len(forced_decisions)) / float(len(all_comment_ids))

    # now look at the proportion forced for the ironic comments
    ironic_comments = get_ironic_comment_ids()
    ironic_ids_str = _make_sql_list_str(ironic_comments)
    forced_ironic_ids =  _grab_single_element(cursor.execute(
                '''select distinct comment_id from irony_label where 
                        forced_decision=1 and comment_id in %s and labeler_id in %s;''' % 
                                (ironic_ids_str, labeler_id_str))) 

    '''
    regression bit.
    '''
    X,y = [],[]
    ### only include those that have been labeled by everyone for now
    #task, tuples, comments_to_summary_lbls, comments_to_labelers = agreement()
    #all_comment_ids = [int(id_) for id_ in comments_to_summary_lbls.keys()]

    for c_id in all_comment_ids:
        if c_id in forced_decisions:
            y.append(1.0)
        else:
            y.append(0.0)

        if c_id in ironic_comments:
            X.append([1.0])
        else:
            X.append([0.0])

    X = sm.add_constant(X, prepend=True)
    #pdb.set_trace()
    logit_mod = sm.Logit(y, X)
    logit_res = logit_mod.fit()
    
    print logit_res.summary()
    return logit_res

def get_ironic_comments(urls_too=False):
    ironic_ids = get_ironic_comment_ids()
    comments = grab_comments(ironic_ids)
    if not urls_too:
        return comments

    urls = []
    for comment_id in ironic_ids:
        urls.append(_grab_single_element(
            cursor.execute("select permalink from irony_comment where id='%s'" % comment_id)))
    #pdb.set_trace()
    return (comments, urls)
        

def get_labeled_comment_breakdown():
    pass

def _get_entries(a_list, indices):
    return [a_list[i] for i in indices]

def get_NNPs_from_comment(comment):
    NNPs = []
    for sentence in nltk.sent_tokenize(comment):
        tokenized_sent = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokenized_sent)
        NNPs.extend([t[0] for t in pos_tags if t[1]=="NNP"])
    return NNPs

def get_all_NNPs():
    comments = get_labeled_thrice_comments()
    comments_str = _make_sql_list_str(comments)
    comments_and_redditors = list(cursor.execute(
                '''select id, redditor, subreddit from irony_comment where id in %s;''' % 
                comments_str))
    
    comment_ids = [comment[0] for comment in comments_and_redditors]
    comment_texts = grab_comments(comment_ids)

    # yes, inefficient
    users = [comment[1] for comment in comments_and_redditors]
    subreddits = [comment[-1] for comment in comments_and_redditors]

    #comment_NNPs = []
    related_subreddit_comments, related_user_comments = [], []
    for i, comment in enumerate(comment_texts[:5]):
        print "on comment %s" % (i+1)
        NNPs = get_NNPs_from_comment(comment)
        subreddit = subreddits[i]
        user = users[i]
        #comment_NNPs.append(NNPs)
        related_subreddit_comments_i, related_user_comments_i = [], []
        for NNP in NNPs:
            related_subreddit_comments_i.extend(
                [c.body for c in search_reddit.search_subreddit_comments(NNP, subreddit)])
            related_user_comments_i.extend(
                [c.body for c in search_reddit.search_comments_by_user(NNP, user)])
        
        related_subreddit_comments.append(related_subreddit_comments_i)
        related_user_comments.append(related_user_comments_i)

    #return comments, comment_NNPs, labels
    pdb.set_trace()


def get_comments_and_labels():
    all_comment_ids = get_labeled_thrice_comments()


    # now look at the proportion forced for the ironic comments
    ironic_comment_ids = get_ironic_comment_ids()
    #ironic_ids_str = _make_sql_list_str(ironic_comments)


    forced_decision_ids = _grab_single_element(cursor.execute(
                '''select distinct comment_id from irony_label where forced_decision=1 and labeler_id in %s;''' % 
                    labeler_id_str)) 

    comment_texts, y = [], []
    for id_ in all_comment_ids:
        comment_texts.append(grab_comments([id_])[0])
        if id_ in ironic_comment_ids:
            y.append(1)
        else:
            y.append(-1)

    return comment_texts, y


def _get_subreddits(comment_ids):
    srs = []
    for c_id in comment_ids:
        sr = _grab_single_element(
            cursor.execute('''select subreddit from irony_comment where id=%s''' % c_id))
        srs.append(sr[0])
    return srs

def get_all_comments_from_subreddit(subreddit):
    #all_comment_ids = get_labeled_thrice_comments()
    #subreddits = _get_subreddits(all_comment_ids)
    #filtered = 
    cursor.execute(
        '''select comment_id from irony_label where 
                comment_id in (select id from irony_comment where subreddit='%s');''' % 
                    subreddit
    )
    subreddit_comments = _grab_single_element(cursor.fetchall())
    return subreddit_comments

### @experimental!
def pretense(just_the_model=False):

    #all_comment_ids = get_labeled_thrice_comments()
    #conservative_comments, liberal_comments = [], []
    #pass
    conservative_comment_ids = get_all_comments_from_subreddit("Conservative")
    conservative_comments = grab_comments(conservative_comment_ids)
    liberal_comment_ids = get_all_comments_from_subreddit("progressive")
    liberal_comments = grab_comments(liberal_comment_ids)
    comment_texts = conservative_comments + liberal_comments
    vectorizer = CountVectorizer(max_features=50000, ngram_range=(1,2), stop_words="english")
    X = vectorizer.fit_transform(comment_texts)
    y = [0 for y_i in xrange(len(conservative_comments))] + [1 for y_i in xrange(len(liberal_comments))]
    #pdb.set_trace()

    #clf = MultinomialNB(alpha=5)
    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(X, y)
    if just_the_model:
        ''' just return the trained progressive/conservative classifier '''
        return clf

    ironic_comment_ids = get_ironic_comment_ids()

    '''
    conservative comments...
    '''
    n_conservative = len(conservative_comment_ids)
    conservative_ironic = [i for i in xrange(n_conservative) if 
                            conservative_comment_ids[i] in ironic_comment_ids]
    X_conservative, y_conservative = [], []
    unironic_conservative_ps, ironic_conservative_ps = [], []
    for i in xrange(n_conservative):
        x_i = X[i]
        p_i = clf.predict_proba(X[i])[0][1]
        X_conservative.append([p_i])

        if i in conservative_ironic:
            ironic_conservative_ps.append(p_i)
            y_conservative.append(1)
        else:
            unironic_conservative_ps.append(p_i)
            y_conservative.append(0)

    X_conservative = sm.add_constant(X_conservative, prepend=True)
    logit_mod = sm.Logit(y_conservative, X_conservative)
    logit_res = logit_mod.fit()
    print " --- conservative ---"
    print logit_res.summary()
    #pdb.set_trace()

    n_liberal = len(liberal_comment_ids)
    liberal_ironic = [i for i in xrange(n_liberal) if 
                            liberal_comment_ids[i] in ironic_comment_ids]

    unironic_liberal_ps, ironic_liberal_ps = [], []
    X_liberal, y_liberal = [], []
    for i in xrange(n_liberal):
        # note the offset
        x_i = X[n_conservative + i]
        p_i = clf.predict_proba(x_i)[0][1]
        X_liberal.append([p_i])
        if i in liberal_ironic:
            ironic_liberal_ps.append(p_i)
            y_liberal.append(1)
        else:
            unironic_liberal_ps.append(p_i)
            y_liberal.append(0)


    X_liberal = sm.add_constant(X_liberal, prepend=True)
    #pdb.set_trace()
    logit_mod = sm.Logit(y_liberal, X_liberal)
    logit_res = logit_mod.fit()
    print "\n --- liberal ---"
    print logit_res.summary()


    ironic_liberal_avg = sum(ironic_liberal_ps)/float(len(ironic_liberal_ps))
    unironic_liberal_avg = sum(unironic_liberal_ps)/float(len(unironic_liberal_ps))
    #pdb.set_trace()

   
def ml_bow():
    all_comment_ids = get_labeled_thrice_comments()
    # now look at the proportion forced for the ironic comments
    ironic_comment_ids = get_ironic_comment_ids()
    #ironic_ids_str = _make_sql_list_str(ironic_comments)

    forced_decision_ids = _grab_single_element(cursor.execute(
                '''select distinct comment_id from irony_label where forced_decision=1 and labeler_id in %s;''' % 
                    labeler_id_str)) 

    comment_texts, y = [], []
    for id_ in all_comment_ids:
        comment_texts.append(grab_comments([id_])[0])
        if id_ in ironic_comment_ids:
            y.append(1)
        else:
            y.append(-1)

    # adding some features here
    emoticon_RE_str = '(?::|;|=)(?:-)?(?:\)|\(|D|P)'
    question_mark_RE_str = '\?'
    exclamation_point_RE_str = '\!'
    # any combination of multiple exclamation points and question marks
    interrobang_RE_str = '[\?\!]{2,}'

    for i, comment in enumerate(comment_texts):
        #pdb.set_trace()
        if len(re.findall(r'%s' % emoticon_RE_str, comment)) > 0:
            comment = comment + " PUNCxEMOTICON"
        if len(re.findall(r'%s' % exclamation_point_RE_str, comment)) > 0:
            comment = comment + " PUNCxEXCLAMATION_POINT"
        if len(re.findall(r'%s' % question_mark_RE_str, comment)) > 0:
            comment = comment + " PUNCxQUESTION_MARK"
        if len(re.findall(r'%s' % interrobang_RE_str, comment)) > 0:
            comment = comment + " PUNCxINTERROBANG"
        
        if any([len(s) > 2 and str.isupper(s) for s in comment.split(" ")]):
            comment = comment + " PUNCxUPPERCASE" 
        
        comment_texts[i] = comment
    # vectorize
    vectorizer = CountVectorizer(max_features=50000, ngram_range=(1,2), binary=True, stop_words="english")
    X = vectorizer.fit_transform(comment_texts)
    kf = KFold(len(y), n_folds=5, shuffle=True)
    X_context, y_mistakes = [], []
    recalls, precisions = [], []
    Fs = []
    top_features = []
    for train, test in kf:
        train_ids = _get_entries(all_comment_ids, train)
        test_ids = _get_entries(all_comment_ids, test)
        y_train = _get_entries(y, train)
        y_test = _get_entries(y, test)

        X_train, X_test = X[train], X[test]
        svm = SGDClassifier(loss="hinge", penalty="l2", class_weight="auto", alpha=.01)
        #pdb.set_trace()
        parameters = {'alpha':[.001, .01,  .1]}
        clf = GridSearchCV(svm, parameters, scoring='f1')
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        #precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(y_test, preds)
        tp, fp, tn, fn = 0,0,0,0
        N = len(preds)

        for i in xrange(N):
            cur_id = test_ids[i]

            if cur_id in forced_decision_ids:
                X_context.append([1])
            else:
                X_context.append([0])

            y_i = y_test[i]
            pred_y_i = preds[i]

            if y_i == 1:
                # ironic
                if pred_y_i == 1:
                    # true positive
                    tp += 1 
                    y_mistakes.append(0)
                else:
                    # false negative
                    fn += 1
                    y_mistakes.append(1)
            else:
                # unironic
                if pred_y_i == -1:
                    # true negative
                    tn += 1
                    y_mistakes.append(0)
                else:
                    # false positive
                    fp += 1
                    y_mistakes.append(1)

        recall = tp/float(tp + fn)
        precision = tp/float(tp + fp)
        recalls.append(recall)
        precisions.append(precision)
        f1 = 2* (precision * recall) / (precision + recall)
        Fs.append(f1)
        #pdb.set_trace()
        #top_features.append(show_most_informative_features(vectorizer, clf))

    X_context = sm.add_constant(X_context, prepend=True)
    #pdb.set_trace()
    logit_mod = sm.Logit(y_mistakes, X_context)
    logit_res = logit_mod.fit()
    
    print logit_res.summary()
    #pdb.set_trace()
    clf = SGDClassifier(loss="hinge", penalty="l2", alpha=.01, class_weight="auto")
    clf.fit(X, y)
    print show_most_informative_features(vectorizer, clf)
    #top_features.append(show_most_informative_features(vectorizer, clf))

    print "recalls:"
    print recalls
    print "precisions:"
    print precisions
    print "F1s:"
    print Fs

def grab_comments(comment_id_list, verbose=False):
    comments_list = []
    for comment_id in comment_id_list:
        cursor.execute("select text from irony_commentsegment where comment_id='%s' order by segment_index" % comment_id)
        segments = _grab_single_element(cursor.fetchall())
        comment = " ".join(segments)
        if verbose:
            print comment
        comments_list.append(comment.encode('utf-8').strip())
    return comments_list

def show_most_informative_features(vectorizer, clf, n=100):
    c_f = sorted(zip(clf.coef_[0], vectorizer.get_feature_names()))
    top = zip(c_f[:n], c_f[:-(n+1):-1])
    out_str = []
    for (c1,f1),(c2,f2) in top:
        out_str.append("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1,f1,c2,f2))
    return "\n".join(out_str)



