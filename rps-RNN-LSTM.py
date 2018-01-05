# LSTM for predicting Rock Paper Sessiors

import numpy as np
import random
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
style.use('ggplot')

import time
#  ------------------------------------------ function section ------------------------------------------

# load doc into memory
def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

# setup player2 as random as an alternative to AI
def player2_random_gen(seqlength):
	player2_list_quant = []
	for i in range(seqlength):
		a = random.randint(1,3)		# uniform selection
		if a == 1:					# cutoff point for value below -1 
			a_quant = "r"			# value 1 = ROCK
		elif a == 2:				# cutoff point for value above +1
			a_quant = "p"			# value 3 = SESSIORS
		else:						# everything else in between
			a_quant = "s"			# value 2 =  PAPER
		player2_list_quant.append(a_quant)
	print ('Player2 RPS prediction ditribution')
	print ('Player2 rock:', player2_list_quant.count("r"))	# count the num of ones in list (rock)
	print ('Player2 paper:', player2_list_quant.count("p"))	# count the num of twos in list (paper)
	print ('Player2 sessior:', player2_list_quant.count("s"))	# count the num of threes in list (sessiors)
	with open("player2_rps_UniformRandom.txt","w") as f :	
		for row in player2_list_quant:
			f.write(row)
	return player2_list_quant

# convert raw sequence to one shot array
def one_shot_rps(dataset):
	one_shot = []
	for i in range(len(dataset)):
		if dataset[i] == 'r':
			one_shot.append([1,0,0])
		elif dataset[i] == 'p':
			one_shot.append([0,1,0])
		else:
			one_shot.append([0,0,1])
	one_shot_array = np.array(one_shot)
	print('shape of one short sequence array is:',one_shot_array.shape)
	return one_shot_array

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), :]
		dataX.append(a)
		dataY.append(dataset[i + look_back, :])
	return np.array(dataX), np.array(dataY)

#------------------------------------------config section ------------------------------------------------------------------
play_AI = True				# player 2 is LSTM
onlineMode = False 			# emulate online self-learning mode - use training phase as test reference; for batch size to 1
stackLSTM = False			# stack up multple layers of LSTM
output_dim = 3				# fix this at 3 for RPS use case
input_dim = 3				# fix this at 3 for RPS use case
timestep_length = 200		# play with this for number of hidden nodes in LSTM
training_pct = 0.3			# percentage of overall dataset used for training
dropout = 0.3				# dropout rate (1 means full drop out)
epochs = 25					# num of epochs (one epoch is one sweep of full training set)
hiddenUnits = 50			# size of the hidden units in each cell
validation_split = 0.2		# percentage split in model.fit for built in validation during training phase
np.random.seed(7)			# fix a random seed
# num of training samples submitted for one fwd/bwd pass before weights are updated
batch_size = 1 if onlineMode else 4   

now = time.strftime("%g%m%d%H%M%S")	# date and time code for adding sub directory for tensorboard

# ---------------------------------------load player 1 play sequence-----------------------------------

#raw_rps_seq = load_doc("player1_rps_PRNG.txt")					# long psuedo random generation w distribution
#raw_rps_seq = load_doc("player1_rps_RandRepeat.txt")			# random and deterministic repeated sequence 
#raw_rps_seq = load_doc("player1_rps_LFSR.txt")					# linear feedback shift register as PRNG
raw_rps_seq = load_doc("player1_rps_RANDU.txt")				# RANDU algorithm as PRNG
#raw_rps_seq = load_doc("player1_rps_p1_RawRoshambo.txt")		# player 1 dataset from Roshambo human dataset
#raw_rps_seq = load_doc("player1_rps_p2_RawRoshambo.txt")		# player 2 dataset from Roshambo human dataset

# ----------------------------------------main program, first prep the input data array -------------
print ('---- GENERAL INFO ----')
print ('Player 2 is AI:', play_AI)
print ('size of raw input sequence:', len(raw_rps_seq))
print ('training percentage:', training_pct*100)
print ('timestep_length:', timestep_length)

# split into train and test sets
one_shot_seq = one_shot_rps(raw_rps_seq)
train_size = int(len(one_shot_seq) * training_pct)
test_size = len(one_shot_seq) - train_size
train = one_shot_seq[0:train_size,:]
test = one_shot_seq[train_size:len(one_shot_seq),:]
print ('training dataset size:', train_size)
print ('testing dataset size :', test_size)

# create the sequence of time series arrange by timestep (act like a shift register)
# look_back/timestep dictates the hidden layer; can cause overfitting error when it's too large
trainX, trainY = create_dataset(train, timestep_length)
testX, testY = create_dataset(test, timestep_length)

print ('---- TRAIN SET ----')
print ('training array shape:', trainX.shape)
print ('training label shape:', trainY.shape)
print ('---- TEST SET ----')
print ('testing array shape:', testX.shape)
print ('testing label shape:', testY.shape)

# ---------------------------------create and fit the LSTM network---------------------------------------

# single hidden layer (timestep is one) and cell states
# The "unit" value in this case is the size of the cell state and the size of the hidden state bei
# return sequences is True if stack LSTM architecture
model = Sequential()
model.add(LSTM(hiddenUnits, return_sequences=stackLSTM, input_shape=(timestep_length,input_dim))) 
if stackLSTM:
	model.add(Dropout(dropout))
	model.add(LSTM(hiddenUnits))
model.add(Dropout(dropout))								# stack LSTM architecture;  does not do much in this case	 	
model.add(Dense(output_dim, activation='softmax'))			# the number of unit needs to match the output dim size
print(model.summary())
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the model ; tensorboard will require validation data (or created by split)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/'+now, histogram_freq=1,  write_graph=True, write_grads=True, write_images=False)
model.fit(
	trainX, 
	trainY, 
	epochs=epochs, 
	batch_size=batch_size, 
	verbose=2,
	validation_split = validation_split, 
	callbacks = [tbCallBack])
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# turn the predict probability into one shot prediction; prepare for performance evaluation
trainPredictOneShot=np.zeros(trainPredict.shape, dtype = np.int)
for index, i in enumerate(trainPredict.argmax(axis=1)):
	trainPredictOneShot[index,i] = 1
testPredictOneShot=np.zeros(testPredict.shape, dtype = np.int)
for index, i in enumerate(testPredict.argmax(axis=1)):
	testPredictOneShot[index,i] = 1
# choose online mode for self-learning or non online mode for traditional supervised learning
if onlineMode :
	PredictResult, PredictShape = trainPredictOneShot, trainPredict.shape[0]
	groundtruth = trainY
else: 
	PredictResult, PredictShape = testPredictOneShot, testPredict.shape[0]
	groundtruth = testY

# --------------------------------prediction performance comparison-----------------------------------------
#------------------------------------------------------------------
# player1-move  one-shot-index   |  playerAI-move  one-shot-index |
#------------------------------------------------------------------
# rock          0                |  paper          1              |
# paper         1                |  sessiors       2              |
# sessiors      2                |  rock           0              |
#------------------------------------------------------------------

# choose to use AI or randomm as player2
if not play_AI:														# generate a player 2 move sequence based on random 
	PredictResult = one_shot_rps(player2_random_gen(PredictShape)) 	# using the same length as test adjusted by the timesteps; use testPredict as reference

# see if prediction match
# record who win each round
AI_match=np.zeros(PredictShape, dtype = np.int)				# AI_match init as array of zeros
AI_win, AI_move_list = [], []
for index, predict in enumerate(PredictResult):
	#print ('predict and index is:', predict.argmax(axis=0), index)
	#print ('testY and index is:', testY[index].argmax(axis=0), index)
	AI_move = (predict.argmax(axis=0) + 1) % 3						# calculate the AI opposing move
	AI_move_list.append(AI_move)
	if predict.argmax(axis=0) == groundtruth[index].argmax(axis=0): # check if positive prediction
		AI_match[index] = 1											# record the true positive
		AI_win.append(1)											# record a win if there is predict match since AI will put out winning move
	elif AI_move == groundtruth[index].argmax(axis=0):				# AI move is equal to the player move
		AI_win.append(0)											# record a tie for AI
	else:															# else it is a lost
		AI_win.append(-1) 											# record a lost for AI

print ('Last 20 moves of player 1 versus player 2')
print ('0 = rock, 1 = paper, 2 = sciessors')
print('player 1:', list(groundtruth[-20:-1].argmax(axis=1)))		# print last 20 moves of player 1
print('player 2:', AI_move_list[-20:-1])							# print last 20 moves of player 2

AI_match_rate, sum = [] , 0
for index, i in enumerate(AI_match):
	sum += i
	AI_match_rate.append(sum/(index+1))

AI_win_rate, AI_tie_rate, AI_loss_rate= [], [], []
for index, i in enumerate(AI_win):
	AI_win_rate.append(AI_win[:index].count(1)/(index+1))			# count number of wins thus far divided by total
	AI_tie_rate.append(AI_win[:index].count(0)/(index+1))			# count number of ties thus far divided by total
	AI_loss_rate.append(AI_win[:index].count(-1)/(index+1))			# count number of lost thus far divided by total

#-------------------------------------- plot the result ----------------------------------------------------------------------

#plt.set_xlabel ('Time steps', weight='bold')
#plt.set_ylabel ('Win/Tie/Lost Rate', weight='bold')
plt.title('AI win-tie-loss rate over time', loc='center', weight='bold', color='Black')
plt.plot(AI_match_rate, color='blue')
plt.plot(AI_win_rate, color='green')
plt.plot(AI_tie_rate, color='black')
plt.plot(AI_loss_rate, color='red')
plt.show()