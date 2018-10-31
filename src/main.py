# For Saving plots during a remote connection
import matplotlib
matplotlib.use('Agg')

# Implemented APIs
from database import Database
from pre_processor import Pre_processor
from mirror import Mirror
from elm import Elm
from naive_bayes import Naive_bayes

# Libraries
from time import time

class Main:
	@staticmethod
	def main():
		# Get Data
		train_x, train_y = Database.get_train_data("../dataset/train.csv")
		test_x_raw = Database.get_test_data("../dataset/test.csv")

		# Normalize Data
		train_x = Pre_processor.normalize_data(train_x)
		test_x = Pre_processor.normalize_data(test_x_raw)

		Mirror.plot_heatmap_corr_coef(train_x, "corr_coeff_features.png")

start_time = time()
Main.main() 
end_time = time()
elapsed_time = end_time - start_time
print("Processed in " + str(elapsed_time) + " seconds")

