# For Saving plots during a remote connection
import matplotlib
matplotlib.use('Agg')

# Implemented APIs
from database import Database
from pre_processor import Pre_processor
from mirror import Mirror
from nn import Nn
#from elm import Elm
from naive_bayes import Naive_bayes

# Libraries
from time import time

class Main:
	@staticmethod
	def main():
		# Get Data
		train_x, train_y = Database.get_train_data("../dataset/train.csv")
		test_x_raw = Database.get_test_data("../dataset/test.csv")

		# features_to_ignore: 1-indexing
		features_to_ignore_ = [14, 17, 24, 29, 31, 33, 63, 74, 82, 75, 94, 71, 127, 11, 26, 109]
		train_x = Pre_processor.reduce_features(train_x, [], features_to_ignore_, one_indexing=True, print_str="train_x")
		test_x_raw = Pre_processor.reduce_features(test_x_raw, [], features_to_ignore_, one_indexing=True, print_str="test_x")

		# Normalize Data
		train_x = Pre_processor.normalize_data(train_x)
		test_x = Pre_processor.normalize_data(test_x_raw)	

start_time = time()
Main.main() 
end_time = time()
elapsed_time = end_time - start_time
print("Processed in " + str(elapsed_time) + " seconds")

