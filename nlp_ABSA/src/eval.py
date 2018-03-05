

def load_label_output(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        return [line.strip().split("\t")[0] for line in f if line.strip()]

def eval_list(glabels, slabels):
    if (len(glabels) != len(slabels)):
        print("\nWARNING: label count in system output (%d) is different from gold label count (%d)\n" % (
        len(slabels), len(glabels)))
    n = min(len(slabels), len(glabels))
    incorrect_count = 0
    for i in range(0, n):
        if slabels[i] != glabels[i]: incorrect_count += 1
    acc = (n - incorrect_count) / n
    print("\nACCURACY: %.2f" % (acc * 100))


def eval_file(goldfile, predfile):
    glabels = load_label_output(goldfile)
    slabels = load_label_output(predfile)
    eval_list(glabels, slabels)

if __name__ == "__main__":
    # Just for testing
    gfilename = "../data/tpabsadataset.csv"
    sfilename = "../data/sysoutput.csv"
    sfilename = "../data/tpabsadataset.csv"
    eval(gfilename, sfilename)




