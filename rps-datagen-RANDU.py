import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


class Randu(random.Random):
    """
    Implementation of the flawed pseudorandom number generating algorithm RANDU.
    See, for more information: http://en.wikipedia.org/wiki/RANDU
    "...its very name RANDU is enough to bring dismay into the eyes and stomachs
    of many computer scientists!"
       -- Donald E. Knuth, The Art of Computer Programming
    """

    def __init__(self, seed=[]):
        try:
            self.seed(seed)
        except TypeError:  # not hashable
            self._state = 1

    def seed(self, x):
        self._state = hash(x) % 0x80000000

    def getstate(self):
        return self._state

    def setstate(self, state):
        self._state = state

    def random(self):
        self._state = (65539 * self._state) % 0x80000000
        return self._state / float(0x80000000)

    @staticmethod
    def check():
        """
        Check against Wikipedia's listed sequence of numbers (start and end of
        the sequence with initial seed 1):
        1, 65539, 393225, 1769499, 7077969, 26542323, ..., 2141591611,
        388843697, 238606867, 79531577, 477211307, 1
        """
        randu = Randu(2141591611)
        actual = []
        for x in range(11):
            actual.append(randu.getstate())
            randu.random()
        assert actual == [2141591611, 388843697, 238606867, 79531577, 477211307,
            1, 65539, 393225, 1769499, 7077969, 26542323]


# Output data for Kaggle competition
if __name__ == '__main__':
    Randu.check()

    # Gosh, I don't know what to seed it with
    randu = Randu(random.randint(1, 0x7FFFFFFF))

    # Gleaned from painstaking analysis of the test data
    upper_bound = 1000000000
    randulist, movelist = [], []
    randulistlength = 5000 
    cutoff1 = int(upper_bound / 3)
    cutoff2 = int(upper_bound * 2 / 3)

    with open('player1_rps_RANDU.txt', 'w') as f:
        for count in range(randulistlength):
            randuvalue = int(randu.random()*upper_bound)
            # print(randuvalue)
            randulist.append(randuvalue)
            if randuvalue <= cutoff1:            # cutoff point for value 
                move = "r"           # value 1 = ROCK
            elif randuvalue>= cutoff2:          # cutoff point for value 
                move = "p"           # value 3 = SESSIORS
            else:                       # everything else in between
                move = "s"           # value 2 =  PAPER
            movelist.append(move)
            f.write(move)

    print ('Player1 RPS RANDU ditribution')
    print ('Player1 rock:', movelist.count("r"))  # count the num of ones in list (rock)
    print ('Player1 paper:', movelist.count("p")) # count the num of twos in list (paper)
    print ('Player1 sessios:', movelist.count("s"))   # count the num of threes in list (sessiors)
    print ('total moves:', len(movelist))

    x = np.array(randulist)
    nbins = 20
    n, bins = np.histogram(x, nbins, density=1)
    pdfx = np.zeros(n.size)
    pdfy = np.zeros(n.size)
    for k in range(n.size):
        pdfx[k] = 0.5*(bins[k]+bins[k+1])
        pdfy[k] = n[k]
    plt.plot(pdfx, pdfy)        # plot the probability distributed function
    plt.show()

