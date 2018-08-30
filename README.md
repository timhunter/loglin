# loglin

Simple trainer for locally-normalized log-linear PCFGs. Uses scipy's BFGS implementation to choose parameters with optimal fit to some training data provided by an input file.

Some sample input files are provided in the examples directory. Run them like this:

    python opt.py examples/argument-adjunct.txt

    python opt.py examples/hunter-dyer-2013.txt

## Regularization

By default it maximizes pure likelihood. To include L2 regularization to discourage weights that are a long way from zero, specify a value for the parameter lambda on the command line using the `-l` flag:

    python opt.py examples/hunter-dyer-2013.txt -l 3

More specifically, the objective function that's maximized is this:

![objective function including regularization](https://latex.codecogs.com/png.latex?\text{objective}(v)&space;=&space;\text{log-likelihood}(\text{training&space;data},&space;v)&space;-&space;\frac{\lambda}{2}&space;||v||^2)

So specifying the value zero is equivalent to not including this flag, i.e. pure likelihood with no regularization, and larger values for lambda produce a stronger bias towards weight values of zero.
