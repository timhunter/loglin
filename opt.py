#!/usr/bin/env python

from __future__ import print_function   # To allow use of python3-style 'print' in python2

import sys
import argparse
import re
from collections import defaultdict
import numpy as np
import scipy.optimize

######################################################################################

class ParseError(Exception):
    pass

class FeatureMismatchError(Exception):
    pass

class DimensionMismatchError(Exception):
    pass

class ProbabilityZeroError(Exception):
    pass

######################################################################################
### Functions for reading training data and model specification from file

token_re = r'(?:\([^\(\)]*\)|\S)+'

def parse(line):
    try:
        [freq, lhs, rhs, feats] = re.split('\s*\|\s*', line)
    except ValueError:
        raise ParseError("Wrong number of fields")
    try:
        freq = int(freq)
    except ValueError:
        raise ParseError("Not an integer value")
    assert re.match('^'+token_re+'$', lhs), ("Left hand side is not a valid token: %s" % lhs)
    rhs_tokens = tuple(re.findall(token_re, rhs))
    feats_tokens = feats.split()
    try:
        feats_vals = list(map(float, feats_tokens))
    except ValueError:
        raise ParseError("Not a float value")
    return (freq, lhs, rhs_tokens, np.array(feats_vals))

def read_input(filename):
    with open(filename, 'r') as fp:
        (line, linenum) = (fp.readline(), 1)
        while line:
            line = line.strip()
            if line != "" and not re.match(r'^#', line):
                try:
                    data = parse(line)
                    yield data
                except ParseError:
                    print("Parse error on line %d" % linenum, file=sys.stderr)
                    raise
            (line, linenum) = (fp.readline(), linenum+1)

# extract_training_data returns a mapping from LHSs to mappings from RHSs to frequencies
def extract_training_data(inp):
    td = defaultdict(lambda: defaultdict(lambda: 0))
    for (freq, lhs, rhs) in inp:
        td[lhs][rhs] += freq
    return td

######################################################################################
### For callbacks from scipy's optimization function

class ProgressStats:
    def __init__(self):
        self.num_iterations = 0
        self.last_objfnval = None

def report_optimization_progress(stats, outputfn, objfnval, weights, freq):
    if freq is not None:
        if stats.last_objfnval is None:
            ftol_check = None
        else:
            ftol_check = (stats.last_objfnval - objfnval) / max([abs(stats.last_objfnval), abs(objfnval), 1])
        if stats.num_iterations % freq == 0:
            outputfn("    iteration: %d    \tfn value: %s    \tftol check: %s" % (stats.num_iterations, objfnval, ftol_check))
        stats.last_objfnval = objfnval
        stats.num_iterations += 1
    return False  # returning True terminates

######################################################################################
### Class for log-linear models

# This is intended as an abstract base class
class LogLinModel():

    def __init__(self):
        print("WARNING: Constructor for LogLinModel -- this is intended as abstract base class, are you sure this is what you want?", file=sys.stderr)

    def score(self, weights, x, y):
        return self.featvec_dot(x, y, weights)

    # Provided as a hook for specialized implementations to override
    def featvec_dot(self, x, y, othervec):
        return np.dot(othervec, self.featvec(x,y))

    # Provided as a hook for specialized implementations to override
    def initial_weights_guess(self, td):
        return np.zeros(self.dim())

    def probs_from_model(self, weights):
        probtable = {}
        for x in self.lhss():
            probtable[x] = {}
            ys = self.rhss(x)
            scores = np.array([self.score(weights,x,y) for y in ys])
            # Using the exp-normalize trick from here: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
            b = scores.max()
            shifted_exp_scores = np.exp(scores - b)
            normalized_probs = shifted_exp_scores / shifted_exp_scores.sum()
            for (y, p) in zip(ys, normalized_probs):
                probtable[x][y] = p
        return probtable

    def train(self, td, regularization_lambda, initial=None, method=None, outputfn=print, progress_report_freq=None):
        if initial is None:
            initial = self.initial_weights_guess(td)
        outputfn("Initial weight vector is " + ("non-zero" if initial.any() else "zero"))
        if method is None:
            method = 'L-BFGS-B'
        objective = lambda weights: penalty(regularization_lambda, weights) - self.loglikelihood(td, weights)
        gradient = lambda weights: penaltygrad(regularization_lambda, weights) - self.loglhdgrad(td, weights)
        stats = ProgressStats()
        callback = lambda w: report_optimization_progress(stats, outputfn, objective(w), w, progress_report_freq)
        res = scipy.optimize.minimize(objective, initial, jac=gradient, method=method, callback=callback)
        outputfn("Optimization results:")
        outputfn("         Function value:       %f" % res.fun)
        outputfn("         Iterations:           %d" % res.nit)
        outputfn("         Function evaluations: %d" % res.nfev)
        try:
            outputfn("         Gradient evaluations: %d" % res.njev)
        except AttributeError:
            pass    # some methods, e.g. L-BFGS-B, don't provide njev
        outputfn("         %s" % res.message)
        return res.x

    def loglikelihood(self, td, weights):
        probtable = self.probs_from_model(weights)
        def logprob(x,y):
            prob = probtable[x][y]
            if prob == 0.0:
                raise ProbabilityZeroError
            else:
                return np.log(prob)
        try:
            ll = 0
            for (lhs,d) in td.items():
                ll += sum([ freq * logprob(lhs,rhs) for (rhs,freq) in d.items() ])
        except ProbabilityZeroError:
            ll = np.finfo('float').min  # the smallest number possible
        return ll

    # This function corresponds to equation (6) in Mike Collins' notes here:
    #       http://www.cs.columbia.edu/~mcollins/loglinear.pdf
    # First note that the right-hand side of equation(6):
    #       sum_i [ f_k(x_i, y_i) ]  -  sum_i sum_y [ p(y, x_i, v) f_k(x_i, y) ]
    # can be re-expressed as
    #       sum_i [ f_k(x_i, y_i)  -  sum_y [ p(y, x_i, v) f_k(x_i, y) ] ]
    # This determines a number for each value of k; or, a vector with one entry 
    # for each value of k. That vector is what this function computes. 
    # The subtraction can be thought of as subtracting the expected contribution 
    # of the kth feature from the observed contribution.
    def loglhdgrad(self, td, weights):

        # Precompute p(y,x,v) for each (x,y) pair
        probtable = self.probs_from_model(weights)

        # For each possible x/lhs, we first compute the expectation component 
        # (which does not depend on the observed values of y/rhs), and then we 
        # calculate the contribution of each training item. Our 'freq' does not 
        # appear in Collins' equation; it's the number of times this particular 
        # (x,y) pair shows up in Collins' i-indexed training set.
        result = np.zeros(self.dim())
        for (lhs,d) in td.items():
            expectation = np.zeros(self.dim())
            for rhsp in self.rhss(lhs):
                expectation += probtable[lhs][rhsp] * self.featvec(lhs,rhsp)
            for (rhs,freq) in d.items():
                result += freq * (self.featvec(lhs,rhs) - expectation)

        assert isinstance(result, np.ndarray) and result.ndim == 1
        return result

    def report_model(self, weights):
        probtable = self.probs_from_model(weights)
        print("######################################")
        for lhs in self.lhss():
            for rhs in sorted(self.rhss(lhs)):
                print("%12.6f\t%.6f\t%s --> %s" % (self.score(weights,lhs,rhs), probtable[lhs][rhs], lhs, " ".join(rhs)))
        print("######################################")

class LogLinModelMixed(LogLinModel):

    # An indicator group is a pair (f,ks) which represents a collection of features [(lambda (x,y): 1 if f(x,y) == k else 0) for k in ks]
    def __init__(self, rulelist, featfuncs, indicator_groups):
        self._featfuncs = featfuncs
        self._featvecdict = {}

        self._ruledict = defaultdict(lambda: [])
        for (x,y) in rulelist:
            self._ruledict[x].append(y)

        # For each indicator group, we store a triple consisting of:
        #   offset: the number of features, both function features and indicators, that precede this one
        #   f: the function for extracting the class from and (x,y) pair
        #   d: a dictionary mapping each class to its index
        self._indicator_groups = []
        offset = len(featfuncs)
        for (f,ks) in indicator_groups:
            d = {k : i for (i,k) in enumerate(ks)}
            self._indicator_groups.append((offset,f,d))
            offset += len(ks)
        self._dim = offset

    def dim(self):
        return self._dim

    def lhss(self):
        return self._ruledict.keys()

    def rhss(self, x):
        return self._ruledict[x]

    # Construct a weight vector where each indicator feature's weight is the log of its frequency in the training data. 
    # If the model only consists of ``basic'' indicator features, then this should take us straight to the maximum likelihood.
    def initial_weights_guess(self, td):
        v = np.zeros(self.dim())
        for (offset,f,d) in self._indicator_groups:
            for x in td:
                for (y,freq) in td[x].items():
                    v[offset + d[f(x,y)]] += freq
        return np.ma.log(v).filled(0)  # Take the log of entries in v where valid, fill the rest with 0

    # We don't actually memoize a feature vector here, because it's probably huge and sparse. 
    # Instead, we just memoize a sequence of (index,value) pairs representing the non-zero entries in the vector f(lhs,rhs).
    def featvec(self, lhs, rhs):
        try:
            nonzero_entries = self._featvecdict[(lhs,rhs)]
        except KeyError:
            nonzero_entries = [(i, f(lhs,rhs)) for (i,f) in enumerate(self._featfuncs) if f(lhs,rhs) != 0]
            for (offset,f,d) in self._indicator_groups:
                index = offset + d[f(lhs,rhs)]
                nonzero_entries.append((index,1))
            self._featvecdict[(lhs,rhs)] = tuple(nonzero_entries)
        result = np.zeros(self.dim())
        for (index,value) in nonzero_entries:
            result.put(index,value)
        return result

    def featvec_dot(self, x, y, othervec):
        total = 0
        for (i,f) in enumerate(self._featfuncs):
            total += f(x,y) * othervec[i]
        for (offset,f,d) in self._indicator_groups:
            total += othervec[offset + d[f(x,y)]]
        return total

class LogLinModelWithFunctions(LogLinModelMixed):
    def __init__(self, rulelist, featfuncs):
        super().__init__(rulelist, featfuncs, [])

# Subclass for models with only ``basic'' features, in the sense of Berg-Kirkpatrick et al 2010 p.583, 
# i.e. indicator features that emulate classical generative models
class LogLinModelBasic(LogLinModelMixed):
    def __init__(self, xypairs):
        indicator_group = ((lambda x,y: (x,y)), xypairs)
        super().__init__(xypairs, [], [indicator_group])

######################################################################################
### Regularization/priors; providing L2 regularization as the only option for now

def penalty(lam, weights):
    return (lam / 2) * (np.linalg.norm(weights)**2)

def penaltygrad(lam, weights):
    return lam * weights

######################################################################################

def run(filename, regularization_lambda):

    # This input data contains both rules' feature vectors and their training frequencies together
    input_data = list(read_input(filename))

    # Set up ruletbl as a mapping from LHSs to mappings from RHSs to feature vectors, 
    # and numfeats as the number of parameters
    ruletbl = defaultdict(lambda: {})
    numfeats = None
    for (freq,lhs,rhs,feats) in input_data:
        if numfeats is None:
            numfeats = len(feats)
        else:
            assert numfeats == len(feats), "Mismatching dimensions"
        old_feats = ruletbl[lhs].get(rhs, None)
        if old_feats is not None:
            assert np.array_equal(old_feats, feats), ("Mismatching features for lhs %s and rhs %s" % (lhs,rhs))
        else:
            ruletbl[lhs][rhs] = np.array(feats)

    ######################################################

    xypairs = [(lhs,rhs) for lhs in ruletbl for rhs in ruletbl[lhs].keys()]

    ### Standard basic model for the ``naive parametrization'', with an indicator for each (lhs,rhs) pair
    #m = LogLinModelBasic(xypairs)

    ## Using the generic class with feature functions to implement the model based on feature vectors from the input file
    featfuncs = list(map((lambda i: lambda x,y: ruletbl[x][y][i]), range(numfeats)))
    m = LogLinModelWithFunctions(xypairs, featfuncs)

    ######################################################

    td = extract_training_data([(freq,lhs,rhs) for (freq,lhs,rhs,feats) in input_data])

    # Do the optimization
    weights = m.train(td, regularization_lambda)
    print("Found optimal parameter values:", weights)
    llhd = m.loglikelihood(td, weights)
    penalty_term = penalty(regularization_lambda, weights)
    print("At this point:  penalty - log-likelihood  =  %f - %f  =  %f" % (penalty_term, llhd, penalty_term - llhd))

    # Print out the rules with their optimized probabilities
    m.report_model(weights)

def main(argv):
    argparser = argparse.ArgumentParser()
    argparser.add_argument("training_file", metavar="TRAINING_FILE", type=str, help="File containing training data")
    argparser.add_argument("-l2", dest="l2_lambda", metavar="LAMBDA", type=float, default=0, help="Lambda parameter value for L2 regularization")
    args = argparser.parse_args()
    print("Training data file:", args.training_file, file=sys.stderr)
    print("Regularization parameter lambda:", args.l2_lambda, file=sys.stderr)
    run(args.training_file, args.l2_lambda)
    print("Done, exiting", file=sys.stderr)

if __name__ == "__main__":
    main(sys.argv)

