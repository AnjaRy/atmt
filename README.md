# atmt code base
Materials for the first assignment of "Advanced Techniques of Machine Translation".
Please refer to the assignment sheet for instructions on how to use the toolkit.

The toolkit is based on [this implementation](https://github.com/demelin/nmt_toolkit).


# Environment Setup

### conda

```
# ensure that you have conda (or miniconda) installed (https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and that it is activated

# create clean environment
conda create --name atmt36 python=3.6

# activate the environment
conda activate atmt36

# intall required packages
pip install torch==1.6.0 numpy tqdm sacrebleu
```

### virtualenv

```
# ensure that you have python 3.6 downloaded and installed (https://www.python.org/downloads/)

# install virtualenv
pip install virtualenv

# create a virtual environment named "atmt36"
virtualenv --python=python3 atmt36

# launch the newly created environment
atmt36/bin/activate

# intall required packages
pip install torch==1.6.0 numpy tqdm sacrebleu
```

# Assignments

- [] Assignment 1: Training and evaluating an NMT model
- [] Assignment 2: ??
- [] Assignment 3: ??
- [] Assignment 4: ??
- [] Assignment 5: ??


