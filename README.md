# loglin

Simple trainer for locally-normalized log-linear PCFGs. Uses BFGS to maximize likelihood; no regularization yet.

Some sample input files are provided in the examples directory. Run them like this:

    python opt.py examples/argument-adjunct.txt

    python opt.py examples/hunter-dyer-2013.txt
