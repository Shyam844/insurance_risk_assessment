
from constants import Constants
from database import Database
from pre_processor import Pre_processor
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

class Nn:

	@staticmethod
	def epoch(train_x, train_y, test_x, test_x_raw, neurons, act_func, filename):
		train_y = Pre_processor.one_hot_encoding(train_y)
		model = Sequential()
		model.add(Dense(units=neurons, activation=act_func, input_dim=Constants.tot_features))
		model.add(Dense(units=Constants.tot_labels, activation='sigmoid'))	
		optimizer_ = SGD(lr=0.3)
		model.compile(loss='categorical_crossentropy', optimizer=optimizer_, metrics=['accuracy'])
		model.fit(train_x, train_y, epochs=3, batch_size=30, verbose=2)
		pred_y = model.predict(test_x, batch_size=128)
		pred_y = Pre_processor.one_hot_decoding_full(pred_y)
		Database.save_results(test_x_raw, pred_y, filename)
			
	
