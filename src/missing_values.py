import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
file = pd.read_csv('train.csv', delimiter= ',', header= None)

def classes_data_count():
	lines = len(file[1]) - 1
	mul = 100/float(lines)
	classes = []
	for i in range(1,9):
		classes.append("class" + str(i))
	freq_count = [0 for i in range(1,9)]

	labels = file.iloc[1:, 127]

	for l in labels:
		freq_count[int(l)-1] +=1 
	freq_count = [i*mul for i in freq_count]


	bar_width = 0.50
	y_pos = np.arange(len(classes))
	plt.bar(y_pos, freq_count, bar_width)
	plt.xticks(y_pos , classes)
	plt.xticks(rotation=10)
	plt.title(" % Of classes in total dataset")
	plt.xlabel('classes')
	plt.ylabel('%')
	plt.show()


# print file.isnull().sum()[file.isnull().sum() !=0] 
def missing_features_percentage():
	lines = len(file[1]) - 1
	mul = 100/float(lines)
	arr = [ 'Employment_Info_1', 'Employment_Info_4', 'Employment_Info_6', 'Insurance_History_5', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32' ]
	val = [   19, 6779,
	10854,
	25396,
	28656,
	34241,
	19184,
	41811,
	8889,
	58824,
	44596,
	55580,
	58274 ]
	x_val = []
	for j in val:
		x_val.append( j*mul )
	bar_width = 0.50
	y_pos = np.arange(len(arr))
	plt.bar(y_pos, x_val, bar_width)
	plt.xticks(y_pos , arr)
	plt.xticks(rotation=10)
	plt.title(" % Of values are missing in features")
	plt.xlabel('features name')
	plt.ylabel('%')
	plt.show()










