from nupic.research.TP import TP
import numpy as np

tp = TP(numberOfCols=20, cellsPerColumn=3,
        initialPerm=0.5, connectedPerm=0.5,
        minThreshold=10, newSynapseCount=10,
        permanenceInc=0.1, permanenceDec=0.0,
        activationThreshold=6,
        globalDecay=0, burnIn=1,
        checkSynapseConsistency=False,
        pamLength=10)

list = np.array([[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]])

list1 = np.array([])

for i in range(30):
    for j in range(len(list)):
        tp.compute(list[j], enableLearn=True, computeInfOutput=False)

    tp.reset()

def formatRow(x):
    s = ''
    for c in range(len(x)):
        if c > 0 and c % 10 == 0:
            s += ''
        s += str(x[:])
    s += ''
    return s

for i in range(len(list)):
    print "\n\n-------", "list", i, "--------------"
    tp.compute(list[i], enableLearn=False, computeInfOutput=True)
    print "\nAll the active and predicted cells:"
    tp.printStates(printPrevious=False, printLearnState=False)
    print "\n\nThe following columns are predicted by temporal pool, This"
    print "should correspond to columes in the *next* item in the sequence."
    predictedCells = tp.getPredictedState()
    # print "\n---------------------------------------\n"
    # print predictedCells
    print formatRow(predictedCells.max(axis=1).nonzero())
