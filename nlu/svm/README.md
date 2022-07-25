# SVMNLU
## Setup
1. Upgrade scikit-learn for warning
    ```bash
    $ pip install --upgrade scikit-learn==0.22.0
    ```
2. Fix Classifier
    - Delete following lines in `Classifier.py > classifier() > load()` method for multiprocessing pickle error.
    ```python
    import sys
    import convlab2
    sys.modules['tatk'] = convlab2
    del sys.modules['tatk']
    ```
3. Train SVM model
    ```bash
    $ cd ../../ConvLab-2/convlab2/nlu/svm/multiwoz
    $ python preprocess.py usr
    $ cd ../
    $ PYTHONPATH=../../..
    $ python train.py multiwoz/configs/multiwoz_usr.cfg
    ```