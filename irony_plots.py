import os
import pickle

import pylab
# seaborn makes things pretty.
# http://www.stanford.edu/~mwaskom/software/seaborn/installing.html
import seaborn as sns 
sns.set(style="nogrid")
import pickle # for loading cached (pickled) results
import numpy as np



class ResultPlotter:

    def __init__(self, base_dir="results"):
        self.base_dir = base_dir
        print "trying to load results..."
        try:
            self.load_results()
        except:
            print '''whoops -- failed to unpickle results. are you sure you've 
                     placed them in the %s subdir (and that they have the expected
                     names)?''' % self.base_dir
            

        print "success!"

    def load_results(self):
        '''
        assumes you've pickled and dump the empirical results
        (lists of results) for the respective metrics
        '''
        ### svm baseline
        self.svm_baseline_precisions = self._unpickle("precisions_svm_baseline.pickle")
        self.svm_baseline_recalls = self._unpickle("recalls_svm_baseline.pickle")
        self.svm_baseline_Fs = self._unpickle("Fs_svm_baseline.pickle")

        ### using interaction terms
        self.interaction_precisions = self._unpickle("precisions_interactions.pickle")
        self.interaction_recalls = self._unpickle("recalls_interactions.pickle")
        self.interaction_Fs = self._unpickle("F_interactions.pickle")

        ### sentiment features 
        self.sent_precisions = self._unpickle("precisions_sent.pickle")
        self.sent_recalls = self._unpickle("recalls_sent.pickle")
        self.sent_Fs = self._unpickle("Fs_sent.pickle")

        ### the whole she-bang
        self.all_precisions = self._unpickle("precisions_all.pickle")
        self.all_recalls = self._unpickle("recalls_all.pickle")
        self.all_Fs = self._unpickle("Fs_all.pickle")


    def _unpickle(self, fname):
        return pickle.load(open(os.path.join(self.base_dir, fname)))

    def summary_results_table(self):
        out_str = ["baseline SVM\tinteraction terms\tsentiment features\tinteractions + sentiment"]
        F_row = ["F1"]
        for x in [self.svm_baseline_Fs, self.interaction_Fs, self.sent_Fs, self.all_Fs]:
            F_row.append(self._get_summary_str(x))
        out_str.append("\t".join(F_row))

        sens_row = ["Recall"]
        for x in [self.svm_baseline_recalls, self.interaction_recalls, self.sent_recalls, self.all_recalls]:
            sens_row.append(self._get_summary_str(x))
        out_str.append("\t".join(sens_row))

        prec_row = ["Precision"]
        for x in [self.svm_baseline_precisions, self.interaction_precisions, self.sent_precisions, self.all_precisions]:
            prec_row.append(self._get_summary_str(x))
        out_str.append("\t".join(prec_row))

        return "\n".join(out_str)

    def _get_summary_str(self, x):
        return "%0.3f (%0.3f, %0.3f)" % (np.percentile(x, 50), np.percentile(x, 25), np.percentile(x, 75))

    def pretty_plot_F_score(self):
        pass

    def histo_plots(self):
        pylab.clf()
        colors = sns.color_palette()

        #dn1, bins1, patches1 = pylab.hist(self.interaction_recalls, bins=50, alpha=.4, 
        #                                            color=colors[1], normed=True)
        #pylab.axvline(np.percentile(self.interaction_recalls, 50), color=colors[1], linewidth=10)
        self._add_to_histo(self.svm_baseline_recalls, colors[1])
        self._add_to_histo(self.all_recalls, colors[0])
        pylab.xlim((0,.7))
        pylab.axes().yaxis.set_visible(False)
        pylab.axes().spines['top'].set_visible(False)
        pylab.axes().xaxis.set_ticks_position('bottom')
        pylab.savefig("recall.pdf",bbox_inches='tight')

        pylab.clf()
        self._add_to_histo(self.svm_baseline_precisions, colors[1])
        self._add_to_histo(self.all_precisions, colors[0])
        pylab.axes().yaxis.set_visible(False)
        pylab.axes().spines['top'].set_visible(False)
        pylab.axes().xaxis.set_ticks_position('bottom')
        pylab.xlim((0,.3))
        pylab.savefig("precision.pdf", bbox_inches='tight')
        #pylab.savefig("recall.pdf")


    def _add_to_histo(self, x, color):
        n0, bins0, patches0 = pylab.hist(x, bins=50, alpha=.35, 
                                                    color=color, normed=True)
        pylab.axvline(np.percentile(x, 50), color=color,  linewidth=4)







