import array
from collections import Mapping, defaultdict
import numbers
from operator import itemgetter
import re
import unicodedata
import warnings

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.six.moves import xrange
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.hashing import FeatureHasher
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.externals import six
from sklearn.feature_extraction.text import CountVectorizer

class InteractionTermCountVectorizer(CountVectorizer):
    '''
    Extends CountVectorizer to allow for *interaction* features.
    These can specified either at the document or token level 
    (or some product thereof).
    '''
 
    def _count_vocab(self, raw_documents, fixed_vocab, 
                        prefixes=None, interaction_doc_indices=None, 
                        interaction_terms=None, singleton_doc_features=None):
        """
        Create sparse feature matrix, and vocabulary where fixed_vocab=False

        If prefix is not None, a copy of specified tokens pre-prended with
        prefix will be added to each designated document and token. 

        Documents are specified by interaction_doc_indices. If this is None,
        interaction terms will be added to all docs.

        Tokens are specified by interaction_terms; prefixed copies of the tokens 
        in this list will be added to the corpus. If this is None, interaction
        copies will be added for all tokens.
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict(None)
            vocabulary.default_factory = vocabulary.__len__


        analyze = self.build_analyzer()
        j_indices = _make_int_array()
        indptr = _make_int_array()
        indptr.append(0)
        if prefixes is None:
            prefixes = []

        for doc_i, doc in enumerate(raw_documents):
            if singleton_doc_features is not None:
                features_for_doc = singleton_doc_features[doc_i]
                for f in features_for_doc:
                    j_indices.append(vocabulary["f-%s" % f])

            for feature in analyze(doc):
                try:
                    j_indices.append(vocabulary[feature])

                    # now add interaction terms, if necessary
                    #import pdb; pdb.set_trace()
                    for interaction_index, prefix in enumerate(prefixes):

                        cur_interaction_doc_indices = None
                        if interaction_doc_indices is not None:
                            cur_interaction_doc_indices = interaction_doc_indices[interaction_index]

                        cur_interaction_terms = None
                        if interaction_terms is not None:
                            cur_interaction_terms = interaction_terms[interaction_index]


                        if (cur_interaction_doc_indices is None or doc_i in cur_interaction_doc_indices) and (
                            cur_interaction_terms is None or feature in cur_interaction_terms):
                                j_indices.append(vocabulary["i-%s-%s" % (prefix, feature)])
                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue
            indptr.append(len(j_indices))



        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            #import pdb; pdb.set_trace()
            if not vocabulary:
                raise ValueError("empty vocabulary; perhaps the documents only"
                                 " contain stop words")

        # some Python/Scipy versions won't accept an array.array:
        if j_indices:
            j_indices = np.frombuffer(j_indices, dtype=np.intc)
        else:
            j_indices = np.array([], dtype=np.int32)
        indptr = np.frombuffer(indptr, dtype=np.intc)
        values = np.ones(len(j_indices))

        X = sp.csr_matrix((values, j_indices, indptr),
                          shape=(len(indptr) - 1, len(vocabulary)),
                          dtype=self.dtype)
        X.sum_duplicates()
        return vocabulary, X


    def fit_transform(self, raw_documents, y=None, interaction_prefixes=None,
                        interaction_doc_indices=None, interaction_terms=None,
                        singleton_doc_features=None):
        """Learn the vocabulary dictionary and return the count vectors.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        vectors : array, [n_samples, n_features]
        """
        # We intentionally don't call the transform method to make
        # fit_transform overridable without unwanted side effects in
        # TfidfVectorizer.
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        vocabulary, X = self._count_vocab(raw_documents, self.fixed_vocabulary,
                                            prefixes=interaction_prefixes, 
                                            interaction_doc_indices=interaction_doc_indices,
                                            interaction_terms=interaction_terms, 
                                            singleton_doc_features=singleton_doc_features)
        X = X.tocsc()

        if self.binary:
            X.data.fill(1)

        if not self.fixed_vocabulary:
            X = self._sort_features(X, vocabulary)

            n_doc = X.shape[0]
            max_doc_count = (max_df
                             if isinstance(max_df, numbers.Integral)
                             else int(round(max_df * n_doc)))
            min_doc_count = (min_df
                             if isinstance(min_df, numbers.Integral)
                             else int(round(min_df * n_doc)))
            if max_doc_count < min_doc_count:
                raise ValueError(
                    "max_df corresponds to < documents than min_df")
            X, self.stop_words_ = self._limit_features(X, vocabulary,
                                                       max_doc_count,
                                                       min_doc_count,
                                                       max_features)

            self.vocabulary_ = vocabulary

        return X

    def fit(self, raw_documents, y=None, interaction_prefixes=None, 
                                 interaction_doc_indices=None,
                                 interaction_terms=None,
                                 singleton_doc_features=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.

        Returns
        -------
        self
        """
        self.fit_transform(raw_documents, interaction_prefix=interaction_prefixes, 
                                 interaction_doc_indices=interaction_doc_indices,
                                 interaction_terms=interaction_terms,
                                 singleton_doc_features=singleton_doc_features)
        return self


    def transform(self, X, y=None):
        """Transform a sequence of instances to a scipy.sparse matrix.

        Parameters
        ----------
        X : iterable over raw text documents, length = n_samples
            Samples. Each sample must be a text document (either bytes or
            unicode strings, filen ame or file object depending on the
            constructor argument) which will be tokenized and hashed.

        y : (ignored)

        Returns
        -------
        X : scipy.sparse matrix, shape = (n_samples, self.n_features)
            Feature matrix, for use with estimators or further transformers.

        """
        analyzer = self.build_analyzer()
        X = self._get_hasher().transform(analyzer(doc) for doc in X)
        if self.binary:
            pdb.set_trace()
            X.data.fill(1)
        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)
        return X

    def add_interaction_terms(self, raw_documents, interaction_prefix=None, 
                                 interaction_doc_indices=None,
                                 interaction_terms=None):
        """
        Warning -- this is inefficient!

        this is stupid. we just call fit again!
        """
        self.fit(raw_documents, interaction_prefix=interaction_prefix, 
                    interaction_doc_indices=interaction_doc_indices, 
                    interaction_terms=interaction_terms)


def _make_int_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("i"))
