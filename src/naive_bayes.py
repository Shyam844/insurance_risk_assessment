from database import Database
from pre_processor import Pre_processor
from constants import Constants
from sklearn.naive_bayes import GaussianNB

# CONTAINS ALL STATIC MEMBERS
class Naive_bayes:

	@staticmethod
	def epoch(train_x, train_y, test_x, test_x_raw, filename):
		clf = GaussianNB()
		clf.fit(train_x, train_y)
		pred_y = clf.predict(test_x)
		Database.save_results(test_x_raw, pred_y, filename)

	@staticmethod
	def feature_engineering_pca(train_x, train_y, test_x, test_x_raw):
		print("NB Feature Engineering with PCA...")
		count = 1
		while(count < Constants.tot_features):
			print("Top " + str(count) + " features...")
			train_x_mod = Pre_processor.get_top_k_features(train_x, count)
			filename = "nb_top_" + str(count) + "_features.csv"
			Naive_bayes.epoch(train_x_mod, train_y, test_x, test_x_raw, filename)
			count = count+1
