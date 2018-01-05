import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def lfsr2(seed, taps, nbits):
    sr = seed
   #nbits = 8
    while 1:
        xor = 1
        for t in taps:
            if (sr & (1<<(t-1))) != 0:
                xor ^= 1
        sr = (xor << nbits-1) + (sr >> 1)
        yield xor, sr
        if sr == seed:
            break
# ---------------------------- main program ---------------------------------------------
nbits = 12
datalist, movelist = [], []
cutoff1 = int(2**nbits / 3)
cutoff2 = int(2**nbits * 2 / 3)

# -------------------------- generate the random sequence --------------------------------

for xor, sr in lfsr2(0b11001001, (12,11,10,4,1), nbits):
    lfsr_gen = int(bin(2**nbits+sr)[3:], base=2)
    datalist.append(lfsr_gen)
    print (xor, lfsr_gen)


with open("player1_rps_LFSR.txt","w") as f :   
    for i in datalist:
        move = ["r","p","s"][i % 3] # use a mod 3 to create 3 bins
        #if i <= cutoff1:            # cutoff point for value 
        #   move = "r"           # value 1 = ROCK
        #elif i >= cutoff2:          # cutoff point for value 
        #   move = "p"           # value 3 = SESSIORS
        #else:                       # everything else in between
        #   move = "s"           # value 2 =  PAPER
        movelist.append(move)
        f.write(move)

print ('Player1 RPS LRSR ditribution')
print ('Player1 rock:', movelist.count("r"))  # count the num of ones in list (rock)
print ('Player1 paper:', movelist.count("p")) # count the num of twos in list (paper)
print ('Player1 sessios:', movelist.count("s"))   # count the num of threes in list (sessiors)
print ('total moves:', len(movelist))

#---------------- print the PDF chart --------------------------------

x = np.array(datalist)
nbins = 20
n, bins = np.histogram(x, nbins, density=1)
pdfx = np.zeros(n.size)
pdfy = np.zeros(n.size)
for k in range(n.size):
    pdfx[k] = 0.5*(bins[k]+bins[k+1])
    pdfy[k] = n[k]
plt.plot(pdfx, pdfy)        # plot the probability distributed function
plt.show()
