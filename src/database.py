from pre_processor import Pre_processor
from pandas import read_csv
import pickle
import numpy

# CONTAINS ALL STATIC MEMBERS
class Database:
	
	@staticmethod
	def get_train_data(path):
		# Assumption : The last column in CSV file is the label
		# Assumption : The 3rd column of CSV file has strings
		print("Getting Train Data...")
		data = Database.read_csv_(path)
		rows, cols = data.shape
		x = data[:, :cols-1]	# 'cols-1' is excluded
		y = data[:, cols-1]	# only 'cols-1' column is retrieved
		Pre_processor.handle_str_col(x, 2)
		x = numpy.asarray(x, dtype=numpy.float64)
		y = numpy.asarray(y, dtype=numpy.int)
		print("train_x : " + str(x.shape))
		print("train_y : " + str(y.shape))
		return x, y

	@staticmethod
	def get_test_data(path):
		# Assumption : The 3rd column of CSV file has strings
		print("Getting Test Data...")
		data = Database.read_csv_(path)
		rows, cols = data.shape
		x = data[:, :]
		Pre_processor.handle_str_col(x, 2)
		x = numpy.asarray(x, dtype=numpy.float64)
		print("test_x : " + str(x.shape))
		return x

	@staticmethod
	def read_csv_(path):
		data_frame = read_csv(path)
		Pre_processor.handle_missing_values(data_frame)
		data = data_frame.values
		return data

	@staticmethod
	def save_results(test_x, pred_y, file_name):
		#print("Saving Results...")
		test_x = numpy.asarray(test_x, dtype=numpy.int32)
		pred_y = numpy.asarray(pred_y, dtype=numpy.int32)
		col_1 = test_x[:, 0]
		col_2 = pred_y
		book = open(file_name, "a")
		book.write("Id,Response\n")
		count = 0
		while(count < len(pred_y)):
			val_1 = str(col_1[count])
			val_2 = str(col_2[count])
			book.write(val_1 + "," + val_2 + "\n")
			count = count+1
		book.close()	

	@staticmethod
	def save_model(clf):
		f_ = open("model.pkl", "wb")
		pickle.dump(clf, f_)
		f_.close()
