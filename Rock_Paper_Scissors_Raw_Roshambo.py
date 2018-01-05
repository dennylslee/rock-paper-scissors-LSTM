import csv

datasetLength = 5000

def convert_rps (filename, datalist, length):
	with open(filename, "w") as f:
		for j in datalist[:length]:
			if j == 1:
				f.write('r')
			elif j == 2:
				f.write('p')
			else:
				f.write('s')

roshambo_p1, roshambo_p2 = [], []
roshambo_p1_clean, roshambo_p2_clean = [], []
k = 0

with open("Rock_Paper_Scissors_Raw_Roshambo.csv","r") as f :	
	file = csv.reader(f)		
	for row in file: 	
		if k == 0:			# skip the first row
			k = 1
		else:
			roshambo_p1.append(int(row[2]))
			roshambo_p2.append(int(row[3]))

# print (roshambo_p1[:50])
# print (roshambo_p2[:50])

for i in roshambo_p1:
	if i != 0:
		roshambo_p1_clean.append(i) 
for i in roshambo_p2:
	if i != 0:
		roshambo_p2_clean.append(i)

# print (roshambo_p1_clean[:50])
# print (roshambo_p2_clean[:50])

convert_rps ("player1_rps_p1_RawRoshambo.txt", roshambo_p1_clean, datasetLength)
convert_rps ("player1_rps_p2_RawRoshambo.txt", roshambo_p2_clean, datasetLength)
print ('done!')