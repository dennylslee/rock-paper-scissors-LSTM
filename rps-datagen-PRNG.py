# this file generate the rock-paper-sessiors data
# code is done in python 3.6

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import time


#start = time.time()
player1_list_size = 5000	# the length of the series
norm_mu = 0 				# center the guassian mean at zero
norm_signma = 2				# sigma determine the spread
player1_list = []
player1_list_quant = []

# write the list into file

random.seed(11)

for i in range(player1_list_size):
	a = random.gauss(norm_mu,norm_signma) # use built in gaussian distribution PRNG
	if i <= player1_list_size / 2:	# disturb the random seed half way through intentionally
		if a < -1:				# cutoff point for value below -1 
			a_quant = "r"			# value 1 = ROCK
		elif a > 1:				# cutoff point for value above +1
			a_quant = "s"			# value 3 = SESSIORS
		else:					# everything else in between
			a_quant = "p"			# value 2 =  PAPER
	else:						# neutralize the if statement when not in use
		if a < -1:				# cutoff point for value below -1 
			a_quant = "r"			# value 1 = ROCK
		elif a > 1:				# cutoff point for value above +1
			a_quant = "s"			# value 3 = SESSIORS
		else:					# everything else in between
			a_quant = "p"			# value 2 =  PAPER
	player1_list.append(a)
	player1_list_quant.append(a_quant)
print (player1_list_quant[:50])
print (player1_list_quant.count("r"))	# count the num of ones in list (rock)
print (player1_list_quant.count("p"))	# count the num of twos in list (paper)
print (player1_list_quant.count("s"))	# count the num of threes in list (sessiors)
with open("player1_rps_PRNG.txt","w") as f :	
	for row in player1_list_quant:
		f.write(row)


x = np.array(player1_list)
nbins = 20
n, bins = np.histogram(x, nbins, density=1)
pdfx = np.zeros(n.size)
pdfy = np.zeros(n.size)
for k in range(n.size):
    pdfx[k] = 0.5*(bins[k]+bins[k+1])
    pdfy[k] = n[k]
#done = time.time()
#print (done-start)
#plt.plot(player1_list)
#plt.show()
plt.plot(pdfx, pdfy)		# plot the probability distributed function
plt.show()
