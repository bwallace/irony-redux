import pickle
import os
import pdb

import irony_experiments
import byron_plots
import pylab

import numpy as np

'''

hmmmm maybe also run just NNP interactions (mostly to compare baseline to fancy)

@TODO 
    - write code to count and then to generate sparsity plots 

    - move to cluster 


bow, nnp baseline, nnp, just sentiment, nnp + sentiment / thread, 

the story should go:
    BoW
    overall sentiment 
    NNP/subreddit
    NNP/sentiment/thread 
    NNP/subreddit/sentiment/thread 
    NNP/subreddit/sentiment/thread + overall sentiment 
    NNP/subreddit/sentiment/thread + overall sentiment (normed)    
'''
def run_dev_exps(n_iters=100, base_out="aaai-2015-results"):
    '''
    experiments on *development* dataset (bootstrapped)

    baseline v NNP* features, sentiment, NNP* + sentiment, NNP* + sentiment sparsified.
    '''
    print "baseline / BoW"
    baseline = BoW = irony_experiments.sentence_classification(add_interactions=False, cluster_interactions=False, interaction_baseline=False, add_thread_level_interactions=False, add_sentiment=False, add_interaction_sentiment=False, model="SGD", iters=n_iters)

   
    print "BoW and sent"
    BoW_just_sent = irony_experiments.sentence_classification(add_interactions=False, cluster_interactions=False, interaction_baseline=False, add_thread_level_interactions=False, add_sentiment=True, add_interaction_sentiment=False, model="SGD", iters=n_iters)

    print "BoW/NNP"
    BoW_and_NNP = irony_experiments.sentence_classification(add_interactions=True, cluster_interactions=False, interaction_baseline=True, add_thread_level_interactions=False, add_sentiment=False, add_interaction_sentiment=False, model="SGD", iters=n_iters)

    #print "BoW/NNP+"
    #BoW_and_NNP = irony_experiments.sentence_classification(add_interactions=True, cluster_interactions=False, interaction_baseline=True, add_thread_level_interactions=True, add_sentiment=False, add_interaction_sentiment=False, model="SGD", iters=niters)

    print "BoW/subreddit/NNP"
    BoW_and_sr_NNP = irony_experiments.sentence_classification(add_interactions=True, cluster_interactions=False, interaction_baseline=False, add_thread_level_interactions=False, add_sentiment=False, add_interaction_sentiment=False, model="SGD", iters=n_iters)

    #print "BoW/NNP/l1-l2"
    #BoW_and_NNP_l1l2 = irony_experiments.sentence_classification(add_interactions=True, cluster_interactions=False, interaction_baseline=False, add_thread_level_interactions=False, add_sentiment=False, add_interaction_sentiment=False, model="SGDi", iters=100)

    #print "BoW/NNP x sentiment/thread"
    #BoW_NNP_and_thread_sent_baseline = irony_experiments.sentence_classification(add_interactions=True, cluster_interactions=False, interaction_baseline=False, add_thread_level_interactions=True, add_sentiment=False, add_interaction_sentiment=True, model="SGD", iters=n_iters)


    print "BoW/NNP x sentiment/subreddit/thread interactions"
    BoW_NNP_and_thread_sent = irony_experiments.sentence_classification(add_interactions=True, cluster_interactions=False, interaction_baseline=False, add_thread_level_interactions=True, add_sentiment=False, add_interaction_sentiment=True, model="SGD", iters=n_iters)


    print "BoW/NNP/Thread + overall sent"
    BoW_NNP_and_thread_overall_sent = irony_experiments.sentence_classification(add_interactions=True, cluster_interactions=False, interaction_baseline=False, add_thread_level_interactions=True, add_sentiment=True, add_interaction_sentiment=True, model="SGD", iters=n_iters)

    print "BoW/NNP/Thread interactions + overall sent/l1-l2"
    BoW_NNP_and_thread_overall_sent_l1l2 = irony_experiments.sentence_classification(add_interactions=True, cluster_interactions=False, interaction_baseline=False, add_thread_level_interactions=True, add_sentiment=True, add_interaction_sentiment=True, model="SGDi", iters=n_iters)

    #results = [BoW_NNP_and_thread_sent_baseline, BoW_NNP_and_thread_sent, BoW_just_sent, BoW_NNP_and_thread_sent, BoW_NNP_and_thread_sent_l1l2]
    results_names = ["(overall) sent.", "NNP", r"NNP $\times$ subreddit", r"NNP+ $\times$ sent. $\times$ subreddit", r"NNP+ $\times$ sent. $\times$ subreddit + sent.", r"NNP+ $\times$ sent. $\times$ subreddit + sent. ($\ell_{1}$ $\ell_{2}$)"]
    results = [BoW_just_sent, BoW_and_NNP, BoW_and_sr_NNP, BoW_NNP_and_thread_sent, BoW_NNP_and_thread_overall_sent, BoW_NNP_and_thread_overall_sent_l1l2 ]
    #pdb.set_trace()
    byron_plots.delta_dist_plots(baseline, results, results_names)
    
    results_names.insert(0, "baseline")
    results.insert(0, baseline)

    paths = [os.path.join(base_out, "dev-"+s) for s in results_names]
    save_results(results, paths)

    '''
    paths = [os.path.join(base_out, "dev-"+s) for s in ["baseline", "NNP-baseline", "NNP", "sentiment", "NNP_sentiment", "NNP_sent_l1l2"]]
    results = [baseline, BoW_NNP_and_thread_sent_baseline, BoW_NNP_and_thread_sent, BoW_just_sent, BoW_NNP_and_thread_sent, BoW_NNP_and_thread_sent_l1l2]
    save_results(results, paths)
    '''
    ### also need to dump to text file...
    pylab.savefig("dev-results.pdf")

    pylab.clf()
    ### magic 
    nonzero_counts_l2 = results[-2][-1]
    nonzero_counts_l1_l2 = results[-1][-1]
    byron_plots.sparsity_plots(nonzero_counts_l2, nonzero_counts_l1_l2)
    pylab.savefig("dev-sparsities.pdf")

def save_results(results, paths):
    for result, path in zip(results, paths):
        with open(path, 'wb') as outf:
            pickle.dump(result, outf)


def run_test_exps(n_iters=100):
    '''
    held out data! we run repeats to account for SGD variability  
    '''


    print "baseline/BoW"
    baseline = irony_experiments.sentence_classification_heldout(model="SGD", add_interactions=False,  add_sentiment=False, add_thread_level_interactions=False, add_interaction_sentiment=False, seed=50, n_runs=n_iters)

    print "(overall) sent."
    BoW_just_sent = irony_experiments.sentence_classification_heldout(model="SGD", add_interactions=False,  add_sentiment=True, add_thread_level_interactions=False, add_interaction_sentiment=False, seed=50, n_runs=n_iters)

    print "NNP"
    BoW_and_NNP = irony_experiments.sentence_classification_heldout(model="SGD", add_interactions=True, add_sentiment=False,  interaction_baseline=True, add_thread_level_interactions=False, add_interaction_sentiment=False, seed=50, n_runs=n_iters)

    print "NNP x subreddit"
    BoW_and_sr_NNP = irony_experiments.sentence_classification_heldout(add_interactions=True,  interaction_baseline=False, add_thread_level_interactions=False, add_sentiment=False, add_interaction_sentiment=False, seed=50, model="SGD", n_runs=n_iters)

    print "NNP+ x subreddit x sent"
    BoW_NNP_and_thread_sent = irony_experiments.sentence_classification_heldout(add_interactions=True, interaction_baseline=False, add_thread_level_interactions=True, add_sentiment=False, add_interaction_sentiment=True, seed=50, model="SGD", n_runs=n_iters)

    #print "BoW/NNPxsubreddit/Thread interactions x sent"
    #BoW_NNP_and_thread_sent = irony_experiments.sentence_classification_heldout(model="SGD", add_interactions=True,  add_sentiment=False, add_thread_level_interactions=True, add_interaction_sentiment=True, seed=50, n_runs=n_iters)

    print "BoW/NNP+/Thread + sent"
    #BoW_NNP_and_thread_sent = irony_experiments.sentence_classification(add_interactions=True, cluster_interactions=False, interaction_baseline=False, add_thread_level_interactions=True, add_sentiment=True, add_interaction_sentiment=True, model="SGD", iters=niters)
    BoW_NNP_and_thread_sent_overall = irony_experiments.sentence_classification_heldout(model="SGD", add_interactions=True, add_sentiment=True, interaction_baseline=False, add_thread_level_interactions=True, add_interaction_sentiment=True, seed=50, n_runs=n_iters)


    print "BoW/NNP/Thread interactions/sent/l1-l2"
    #BoW_NNP_and_thread_sent_l1l2 = irony_experiments.sentence_classification(add_interactions=True, cluster_interactions=False, interaction_baseline=False, add_thread_level_interactions=True, add_sentiment=True, add_interaction_sentiment=True, model="SGDi", iters=niters)
    BoW_and_NNP_threadl1l2 = irony_experiments.sentence_classification_heldout(model="SGDi", add_interactions=True,  add_sentiment=True, add_thread_level_interactions=True, add_interaction_sentiment=True, seed=50, n_runs=n_iters)
    
    #print "BoW/NNP/Thread interactions/NO sent/l1-l2"
    #BoW_NNP_and_thread_sent_l1l2 = irony_experiments.sentence_classification(add_interactions=True, cluster_interactions=False, interaction_baseline=False, add_thread_level_interactions=True, add_sentiment=True, add_interaction_sentiment=True, model="SGDi", iters=niters)
    #BoW_and_NNP_threadl1l2 = irony_experiments.sentence_classification_heldout(model="SGDi", add_interactions=True,  add_sentiment=False, add_thread_level_interactions=True, add_interaction_sentiment=True, seed=50, n_runs=n_iters)
    
    results_names = ["baseline", "(overall) sent.", "NNP", r"NNP $\times$ subreddit", r"NNP+ $\times$ sent. $\times$ subreddit", r"NNP+ $\times$ sent. $\times$ subreddit + sent.", r"NNP+ $\times$ sent. $\times$ subreddit + sent. ($\ell_{1}$ $\ell_{2}$)"]
    results = [baseline, BoW_just_sent, BoW_and_NNP, BoW_and_sr_NNP, BoW_NNP_and_thread_sent, BoW_NNP_and_thread_sent_overall, BoW_and_NNP_threadl1l2 ]
    
    #r"NNP \times sent. \times subreddit + sent.", r"NNP \times sentiment \times subreddit + sent. ($\ell_{1}$ $\ell_{2}$)"
    #results_names = ["baseline", "NNP", r"NNP+ $\times$ sent. $\times$ subreddit + sent.", r"NNP+ $\times$ sent. $\times$ subreddit + sent. ($\ell_{1}$ $\ell_{2}$)"]
    #results = [BoW_and_NNP_thread, BoW_and_NNP_threadl1l2 ]
    #pdb.set_trace()
    #byron_plots.delta_dist_plots(baseline, results, results_names, exclude_greater_than=.1)


    ### build table with median and percentiles... 

    latex_str = r'''
        \begin{table*} \centering \small
        \begin{tabular}{l l l}    
             & median recall (std. dev.) & median precision (std. dev.) \\
            \hline'''

    def format_num(num):
        return '{0:.{1}f}'.format(num, 3)

    for name, result in zip(results_names, results):
        recalls, precisions = result[1], result[2]
        #median_recall = '{0:.{1}f}'.format(np.median(recalls), 3)
        #lower_recall, upper_recall = 
       
        recall_str = "%s (%s)" % (
                format_num(np.median(recalls)), format_num(np.std(recalls)))
        precision_str = "%s (%s)" % (
                format_num(np.median(precisions)), format_num(np.std(precisions)))


        #(format_num(np.median(recalls), format_num(np.percentile, 25), )

        table_row = name + " & %s & %s \\\\" % (recall_str, precision_str)
        latex_str += table_row 

    latex_str += "\n"
    latex_str += r''' \end{tabular} \end{table*}'''
    #pdb.set_trace()
    return latex_str 

def feature_weight_tables(l1l2=True, n_runs=100, filter_str=None):
    feature_weights = None
    '''
    if l1l2:
        feature_weights = irony_experiments.sentence_classification_heldout(
            model="SGDi", add_interactions=True,  add_sentiment=True, 
            add_thread_level_interactions=True, add_interaction_sentiment=True, 
            seed=2, n_runs=n_runs, return_ifeature_weights=True)
    else: 
        feature_weights = irony_experiments.sentence_classification_heldout(
            model="SGD", add_interactions=True,  add_sentiment=True, 
            add_thread_level_interactions=True, add_interaction_sentiment=True, 
            seed=2, n_runs=n_runs, return_ifeature_weights=True)
    '''
    really_basic = True
    if really_basic:
        feature_weights = irony_experiments.sentence_classification_heldout(
            model="SGDi", add_interactions=True,  add_sentiment=True, 
            add_thread_level_interactions=False, add_interaction_sentiment=True, 
            seed=2, n_runs=n_runs, return_ifeature_weights=True)

    feature_sds = {}
    feature_scores = {}
    for f, values in feature_weights.items():

        # we add zeros for those cases that the feature was dropped (if any)
        feature_vals = np.concatenate((np.array(values), np.zeros(n_runs-len(values))))
        feature_weights[f] = np.mean(feature_vals)
        feature_sds[f] = np.std(feature_vals)
        feature_scores[f] = feature_weights[f]
        #if feature_weights[f] > 0:
        #    feature_scores[f] = feature_weights[f] - 1.96*feature_sds[f]

    import operator 
    sorted_features = sorted(feature_scores.iteritems(), reverse=True, key=operator.itemgetter(1))
    
    # e.g., filter_str might be "f-" if you only want comment level features
    if filter_str is not None:
        filtered = [f for f in sorted_features if filter_str in f[0]]
        sorted_features = filtered

    pdb.set_trace()

    latex_str = r'''
        \begin{table} \centering \footnotesize
        \begin{tabular}{l l l l}
            \multicolumn{2}{c}{\emph{progressive}} & \multicolumn{2}{c}{\emph{conservative}} \\
            \hline
            \multicolumn{1}{c}{feature} & \multicolumn{1}{c}{weight} & \multicolumn{1}{c}{feature} & \multicolumn{1}{c}{weight} \\
            \hline'''

    latex_str += "\n"

    # just doing i- features for now?
    conservative_features, conservative_weights, \
        progressive_features, progressive_weights = [], [], [], []
    for feature, weight in sorted_features:
        weight = feature_weights[feature]
        #if feature.startswith("i-progressive-"):
        if "progressive-" in feature:
            mean_str = '{0:.{1}f}'.format(weight, 3)
            var_str = '{0:.{1}f}'.format(feature_sds[feature], 3)
            progressive_weights.append("%s (%s)" % (mean_str, var_str))

            feature = feature.replace("i-progressive-NNP-", "")

            feature = feature.replace("f-sentence-progressive-", "")
            feature = feature.replace("f-comment-progressive-", "")
            feature = feature.replace("-positive", " (+)")
            feature = feature.replace("-negative", " (-)")

            progressive_features.append(feature)

        #elif feature.startswith("i-conservative-"):
        elif "conservative-" in feature or "Conservative-" in feature:
            mean_str = '{0:.{1}f}'.format(weight, 3)
            var_str = '{0:.{1}f}'.format(feature_sds[feature], 3)
            conservative_weights.append("%s (%s)" % (mean_str, var_str))

            feature = feature.replace("i-conservative-NNP-", "")
            feature = feature.replace("f-sentence-Conservative-", "")
            feature = feature.replace("f-comment-Conservative-", "")

            feature = feature.replace("-positive", " (+)")
            feature = feature.replace("-negative", " (-)")

            conservative_features.append(feature)

    show_top_n = 10
    for i in xrange(show_top_n):
        latex_str += "%s & %s & %s & %s \\\\" % (
                progressive_features[i], progressive_weights[i],
                conservative_features[i], conservative_weights[i])

    latex_str += "\n"
    #NNP:Christmas & 0.165 & NNP:Obamacare & 0.150 \\
    #NNP:Jesus & 0.5 & NNP:president & .3 \\
    latex_str += r''' \end{tabular} \end{table}'''
    return latex_str
