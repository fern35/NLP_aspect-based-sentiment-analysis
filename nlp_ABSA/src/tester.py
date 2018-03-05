import time
import numpy as np

from src.classifier import Classifier
from src.eval import eval_file, eval_list, load_label_output

def set_reproductible():
    # The below is necessary to have reproducible behavior.
    import random as rn
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(17)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)


if __name__ == "__main__":
    set_reproductible()
    datadir = "../data/"
    trainfile =  datadir + "traindata.csv"
    devfile =  datadir + "devdata.csv"
    testfile =  None
    # Basic checking
    start_time = time.perf_counter()
    classifier = Classifier()
    print("\n")
    # Training
    print("1. Training the classifier...\n")
    classifier.train(trainfile)
    # Evaluation on the dev dataset
    print("\n2. Evaluation on the dev dataset...\n")
    slabels = classifier.predict(devfile)
    glabels = load_label_output(devfile)
    eval_list(glabels, slabels)
    if testfile is not None:
        # Evaluation on the test data
        print("\n3. Evaluation on the test dataset...\n")
        slabels = classifier.predict(testfile)
        glabels = load_label_output(testfile)
        eval_list(glabels, slabels)
    print("\nExec time: %.2f s." % (time.perf_counter()-start_time))




