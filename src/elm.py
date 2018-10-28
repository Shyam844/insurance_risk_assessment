from constants import Constants
from pre_processor import Pre_processor
from database import Database
from hpelm import ELM

# CONTAINS ALL STATIC MEMBERS
class Elm:
	
	@staticmethod
	def epoch(train_x, train_y, test_x, test_x_raw, filename):
		features = train_x.shape[1]
		train_y = Pre_processor.one_hot_encoding(train_y)
		clf = ELM(features, Constants.tot_labels)
		clf.add_neurons(550, "sigm")
		clf.train(train_x, train_y, 'CV', 'OP', 'c', k=10)
		pred_y = clf.predict(test_x)
		pred_y = Pre_processor.one_hot_decoding_full(pred_y)
		Database.save_results(test_x_raw, pred_y, filename)

	@staticmethod
	def feature_engineering_pca(train_x, train_y, test_x, test_x_raw):
		print("ELM Feature Engineering with PCA...")
		count = 1
		while(count < Constants.tot_features):
			print("Top " + str(count) + " features...")
			train_x_mod = Pre_processor.get_top_k_features(train_x, count)
			filename = "elm_top_" + str(count) + "_features.csv"
			Elm.epoch(train_x, train_y, test_x, test_x_raw, filename)
			count = count+1

	@staticmethod
	def tune_elm(train_x, train_y, test_x_raw, test_x, act_funcs, neuron_counts):
		'''
		Assumptions:
		1. NN has only 1 hidden layer
		2. act_funcs: list of distinct activation functions
		3. neuron_counts: list of distinct '# of neurons in the hidden layer'
		'''
		print("Tuning ELM...")
		features = train_x.shape[1]
		train_y = Pre_processor.one_hot_encoding(train_y)
		ind_func = 0
		while(ind_func < len(act_funcs)):
			ind_neuron = 0
			cur_act_func = act_funcs[ind_func]
			while(ind_neuron < len(neuron_counts)):
				cur_neuron_count = neuron_counts[ind_neuron]
				print(cur_act_func + " | " + str(cur_neuron_count) + "...")
				clf = ELM(features, Constants.tot_labels)
				clf.add_neurons(cur_neuron_count, cur_act_func)
				clf.train(train_x, train_y, 'CV', 'OP', 'c', k=10)
				pred_y = clf.predict(test_x)
				pred_y = Pre_processor.one_hot_decoding_full(pred_y)		
				file_name = "submission_" + str(cur_neuron_count) + "_" + cur_act_func + ".csv"
				Database.save_results(test_x_raw, pred_y, file_name)
				ind_neuron = ind_neuron+1
			ind_func = ind_func+1
