#!/usr/bin/env python

import sys
import re
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize

######################################################################################

class ParseError(Exception):
    pass

class FeatureMismatchError(Exception):
    pass

class DimensionMismatchError(Exception):
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
    if not re.match('^'+token_re+'$', lhs):
        raise ParseError("Left hand side is not a valid token: %s" % lhs)
    rhs_tokens = tuple(re.findall(token_re, rhs))
    feats_tokens = feats.split()
    try:
        feats_vals = np.array(map(float, feats_tokens))
    except ValueError:
        raise ParseError("Not a float value")
    return (freq, lhs, rhs_tokens, feats_vals)

def read_input(filename):
    with open(filename, 'r') as fp:
        (line, linenum) = (fp.readline(), 1)
        while line:
            line = line.strip()
            if line != "":
                try:
                    data = parse(line)
                    yield data
                except ParseError:
                    print >>sys.stderr, ("Parse error on line %d" % linenum)
                    raise
            (line, linenum) = (fp.readline(), linenum+1)

# build_model returns a triple (m,td,dim), where
#   - m is a mapping from LHSs to mappings from RHSs to feature vectors
#   - td is a mapping from (LHS,RHS) pairs to frequencies
#   - dim is the number of parameters
def build_model(inp):
    m = defaultdict(lambda: {})
    td = defaultdict(lambda: 0)
    dim = None
    for (freq, lhs, rhs, feats) in inp:
        if dim is None:
            dim = len(feats)
        else:
            if dim != len(feats):
                raise DimensionMismatchError("Mismatching dimensions")
        td[(lhs,rhs)] += freq
        old_feats = m[lhs].get(rhs, None)
        if old_feats is not None:
            if not np.array_equal(old_feats, feats):
                raise FeatureMismatchError("Mismatching features for lhs %s and rhs %s" % (lhs,rhs))
        else:
            m[lhs][rhs] = np.array(feats)
    return (m,td,dim)

######################################################################################
### Workings of the model itself

def loglikelihood(m, td, weights):
    freq = lambda x,y: td[(x,y)]
    logprob = lambda x,y: np.log(p(m,y,x,weights))
    ll = sum([freq(lhs,rhs) * logprob(lhs,rhs) for (lhs,rhs) in td])
    return ll

def loglhdgrad(m, td, weights):
    def f(k,x,y):
        return m[x][y][k]
    def loglhdpartial(k):
        a = sum([freq * f(k,lhs,rhs) for ((lhs,rhs),freq) in td.items()])
        b = sum([freq * sum([ f(k,lhs,rhsp) * p(m,rhsp,lhs,weights) for rhsp in m[lhs].keys() ]) for ((lhs,rhs),freq) in td.items()])
        return a - b
    return np.array([loglhdpartial(i) for i in range(len(weights))])

def p(m,y,x,weights):
    #########################################
    ### Naive implementation
    # result = np.exp(score(m,weights,x,y)) / sum([np.exp(score(m,weights,x,yp)) for yp in m[x].keys()])
    # return result
    #########################################
    ### Using the exp_normalize trick from here: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    scores = np.array([score(m,weights,x,yp) for yp in m[x].keys()])
    result = exp_normalize(score(m,weights,x,y), scores)
    return result
    #########################################

def exp_normalize(x,xs):
    b = xs.max()
    ys = np.exp(xs - b, dtype=np.float128)
    y = np.exp(x - b, dtype=np.float128)
    result = y / ys.sum()
    return result

def score(m,weights,x,y):
    feats = m[x][y]
    score = np.dot(weights, feats)
    return score

######################################################################################

def report_model(m, weights):
    print "######################################"
    for lhs in m:
        for rhs in sorted(m[lhs]):
            print lhs, "-->", " ".join(rhs), "\t", p(m, rhs, lhs, weights), "\t", score(m, weights, lhs, rhs)
    print "######################################"

def run(filename):
    (m,td,dim) = build_model(read_input(filename))
    # print "######################################"
    # for lhs in m:
    #     for rhs in m[lhs]:
    #         print lhs, rhs, m[lhs][rhs], td[(lhs,rhs)]
    # print "######################################"
    objective = lambda weights: 0 - loglikelihood(m, td, weights)
    gradient = lambda weights: 0 - loglhdgrad(m, td, weights)
    #res = minimize(objective, np.array([0.0 for x in range(dim)]), jac=gradient, method='Nelder-Mead', options={'disp':True})
    res = minimize(objective, np.array([0.0 for x in range(dim)]), jac=gradient, method='BFGS', options={'disp':True})
    print "Found optimal parameter values:", res.x
    report_model(m, res.x)

def main(argv):
    try:
        [filename] = argv[1:]
    except ValueError:
        print >>sys.stderr, "Need a filename"
        sys.exit(1)
    run(filename)
    print >>sys.stderr, "Done, exiting"

if __name__ == "__main__":
    main(sys.argv)

