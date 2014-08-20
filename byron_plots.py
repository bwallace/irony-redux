import pylab
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np 
import pdb

sns.set(style="white", palette="muted")

def delta_dist_plots(baseline_results, results, names, 
                        metrics_indices=[1,2], 
                        metrics=["recall", "precision"], bins=10,
                        exclude_greater_than=None):
    n_rows = len(results)
    #n_rows = 2
    f, axes = plt.subplots(n_rows, len(metrics), sharex=True)
    
    sns.despine(left=True)
    for i, res in enumerate(results):
        for j, metric_index in enumerate(metrics_indices):
            if names is not None and j%2==0:
                print i 
                print names
                axes[i,j].yaxis.set_label(names[i])
            deltas = np.array(res[metric_index]) - np.array(baseline_results[metric_index])

            # this is really just because once in a (long) while
            # we get very large or small diffs, which screws up the
            # histogram (since we need to make a new bin for them). 
            # so we always drop outliers in both directions here.
            outlier_indices =  [np.argmin(deltas), np.argmax(deltas)]
            #outliers_indices = np.where(outliers)
            print "---"
            print names[i]
            print metrics[j]
            print "EXCLUDING min: %s; max: %s" % (np.min(deltas), np.max(deltas))
            print "---"

            deltas = np.delete(deltas, outlier_indices)
            if exclude_greater_than:
                # if you do pass something in here,
                # make sure you note this in the write-up!!!
                # note: we are not currently setting this flag
                # for figures in paper!
                deltas1 = [d for d in list(deltas) if -.1 <= d <= .1]

                print "total excluded: %s" % len(deltas1)
                print "< -.1: %s; > .1: %s" % (
                        len([d for d in deltas1 if d < -.1]),
                        len([d for d in deltas1 if d > .1]))
                print "---"
                deltas = np.array(deltas1)
                #indices1 = deltas[deltas < -1*exclude_greater_than]
                #print "excluding %s < %s" %(len(indices1), -1*exclude_greater_than)
                #deltas = np.delete(deltas, indices1)

                #indices2 = deltas[deltas > exclude_greater_than]
                #print "excluding %s > %s" %(len(indices2), exclude_greater_than)
                #deltas = np.delete(deltas, indices2)

            #deltas[deltas < -.1] = 0
            #deltas[deltas > .1] = 0
            #deltas = np.delete(deltas, )

            #pdb.set_trace()
            if j == 0:
                sns.distplot(deltas, ax=axes[i,j], kde=False, bins=bins)
            else:
                sns.distplot(deltas, ax=axes[i,j], kde=False, bins=bins)
            #axes[i,0].set_ylim((0, 7))

            # maybe also add the mean?
            mean= np.median(deltas)
            axes[i,j].axvline(mean, ls="--", linewidth=1.5)
            axes[i,j].axvline(np.mean(deltas), ls="-", linewidth=2, color="blue")
            axes[i,j].axvline(0, linewidth=1.5, color="black")


    axes[0,0].set_xlim((-.159, .159))
    axes[1,0].set_xlim((-.159, .159))
    #axes[1,0].set_xlim((-.05, .05))

    plt.setp(axes, yticks=[])

    name_index = 0
    metrics_index = 0
    for ax_index, ax in enumerate(f.get_axes()):
        if ax_index % 2 == 0:
            #pdb.set_trace()
            ax.set_ylabel(names[name_index], rotation="horizontal", labelpad=50)
            name_index += 1

    for ax_index, ax in enumerate(f.get_axes()[-len(metrics):]):
        ax.set_xlabel(r"$\Delta$ " + metrics[ax_index])
    plt.tight_layout()

def sparsity_plots(nonzero_countl2, nonzero_countl1l2):
    #sns.boxplot([nonzero_countl2, nonzero_countl1l2], names=[r"$\ell_{2}$", r"$\ell_{1}$ $\ell_{2}$" ])
    ax = sns.violinplot([nonzero_countl2, nonzero_countl1l2], names=[r"$\ell_{2}$", r"$\ell_{1}$ $\ell_{2}$" ])
    sns.despine(trim=True);
    ax.set(ylabel="Number of non-zero weights")
