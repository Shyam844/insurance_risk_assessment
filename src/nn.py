
from constants import Constants
from database import Database
from pre_processor import Pre_processor
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

class Nn:

	@staticmethod
	def epoch(train_x, train_y, test_x, test_x_raw, neurons, activation_, epochs_, lr_):
		train_y = Pre_processor.one_hot_encoding(train_y)
		model = Sequential()
		# input, hidden layer
		model.add(Dense(units=neurons, activation=activation_, input_dim=Constants.tot_features))
		# 2nd hidden layer
		model.add(Dense(units=neurons, activation=activation_))
		# Output layer
		model.add(Dense(units=Constants.tot_labels, activation='sigmoid'))	
		optimizer_ = SGD(lr=lr_)
		model.compile(loss='categorical_crossentropy', optimizer=optimizer_, metrics=['accuracy'])
		model.fit(train_x, train_y, epochs=epochs_, batch_size=30, verbose=2)
		pred_y = model.predict(test_x, batch_size=128)
		pred_y = Pre_processor.one_hot_decoding_full(pred_y)
		filename = "nn_2_" + str(neurons) + "_" + activation_ + "_" + str(epochs_) + "_" + str(lr_) + ".csv"
		Database.save_results(test_x_raw, pred_y, filename)
	
	@staticmethod
	def tune(train_x, train_y, test_x, test_x_raw, neurons, activations, epochs, lrs):
		for neuron in neurons:
			for activation in activations:
				for epoch in epochs:
					for lr in lrs:
						print(str(neuron) + " | " + activation + " | " + str(epoch) + " | " + str(lr) + "...")
						Nn.epoch(train_x, train_y, test_x, test_x_raw, neuron, activation, epoch, lr)


	
	
