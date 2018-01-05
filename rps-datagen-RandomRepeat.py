
repeat = 50
randomstr = 'sprrpsrpssprrpsrppsprrspsspprrssssppspspsrspssprpsrsprrspprsppspsprrssprsprspsppss'
with open("player1_rps_RandRepeat.txt","w") as f :	
	for row in range(repeat):
		f.write(randomstr)

print (repeat * randomstr.count("r"))	# count the num of ones in list (rock)
print (repeat * randomstr.count("p"))	# count the num of twos in list (paper)
print (repeat * randomstr.count("s"))	# count the num of threes in list (sessiors)