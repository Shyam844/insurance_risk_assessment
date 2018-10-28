# For Saving plots during a remote connection
import matplotlib
matplotlib.use('Agg')

# Implemented APIs
from database import Database
from pre_processor import Pre_processor
from mirror import Mirror
from elm import Elm

# Libraries
from time import time
from sklearn.naive_bayes import GaussianNB

class Main:

	def main():
		# Get Data
		train_x, train_y = Database.get_train_data("../dataset/train.csv")
		test_x_raw = Database.get_test_data("../dataset/test.csv")

		# Normalize Data
		train_x = Pre_processor.normalize_data(train_x)
		test_x = Pre_processor.normalize_data(test_x_raw)

	'''
		
		#tune_elm(train_x, train_y, test_x_raw, test_x, ["tanh", "sigm"], [11, 12, 13])
		Elm.tune_elm(train_x, train_y, test_x_raw, test_x, ["tanh", "sigm"], [300, 350, 400, 450, 500, 550, 600, 650, 700, 750])

		# Visualize Data
		Mirror.visualize_pca(train_x, train_y)
		Mirror.visualize_tsne(train_x, train_y)

		Pre_processor.feature_mining(train_x, train_y)

		# Train the Model
		clf = GaussianNB()
		print("Training...")
		clf.fit(train_x, train_y)

		# Test the Model
		print("Testing...")
		pred_y = clf.predict(test_x)

		# Save Results
		Database.save_results(test_x_raw, pred_y, "GaussianNB.csv")

		'''

start_time = time()
Main.main()
end_time = time()
elapsed_time = end_time - start_time
print("Processed in " + str(elapsed_time) + " seconds")

