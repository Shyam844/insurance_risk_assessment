Project : https://www.kaggle.com/c/prudential-life-insurance-assessment

Classifier vs QuadraticWeightedKappa
------------------------------------
Naive Bayes: 
	- .290
	- Default parametres
	- 6 Seconds
Logistic Regression: 
	- .289
	- Default Parametres
	- 33 Seconds
SVM: 
	- -0.00010
	- Default Parametres
	- 106 Minutes
	- Always Predicts class 8

KNN:
	- .07
	- n_neighbors=11
	- 25 Seconds
ELM:
	- .4455
	- clf.add_neurons(250, "tanh")	clf.train(train_x, train_y, 'CV', 'OP', 'c', k=10)
	- 85 seconds
	---------------------------
	- .49848
	-	clf = ELM(features, tot_labels)
		clf.add_neurons(500, "sigm")
		clf.add_neurons(250, "tanh")
		clf.add_neurons(150, "tanh")
		clf.add_neurons(1000, "sigm")
		clf.add_neurons(500, "tanh")
		clf.add_neurons(250, "sigm")
		clf.add_neurons(50, "tanh")
		clf.train(train_x, train_y, 'CV', 'OP', 'c', k=10)
	- almost 5 hours
	----------------------------
	- .363
	- 	clf.add_neurons(68, "tanh")
		clf.train(train_x, train_y, 'CV', 'OP', 'c', k=10)
	- 42 seconds
	- https://stats.stackexchange.com/a/1097
	----------------------------
	-.427
	- 	clf.add_neurons(220, "tanh")
		clf.train(train_x, train_y, 'CV', 'OP', 'c', k=10)
	- 304 seconds
	- https://stats.stackexchange.com/a/136542
	----------------------------
	-.45
	- 	clf.add_neurons(220, "sigm")
		clf.train(train_x, train_y, 'CV', 'OP', 'c', k=10)
	- 312 seconds


Tutorials:
-----------
Missing Values: https://machinelearningmastery.com/handle-missing-data-python/

To-do
-----
ELM with an argument - accelerator="GPU" to use the GPU; install skcuda on
compute node for this
