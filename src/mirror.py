from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pandas import read_csv
import seaborn
import numpy

# CONTAINS ALL STATIC MEMBERS
class Mirror:

	@staticmethod
	def plot_heatmap_corr_coef(data, filename):
		data = numpy.corrcoef(data, rowvar=False)
		axes = seaborn.heatmap(data, vmin=-1, vmax=1)
		axes.set_xlabel("Features")
		axes.set_ylabel("Features")
		axes.set_title("Correlation Coefficient of Features")
		fig = axes.get_figure()
		fig.set_size_inches(11.7, 8.27)
		fig.savefig(filename, dpi=100)

	@staticmethod
	def one_d(points, labels, path):
		pyplot.scatter(points, labels, c=labels, s=1)
		pyplot.savefig(path)		

	@staticmethod
	def two_d_from_csv(csv, png):
		# Assumtion: 2nd column is the only data that is considered
		data = Mirror.read_csv_(csv)
		x = data[:, 0]
		y = data[:, 1]
		pyplot.plot(x, y)
		pyplot.xlabel("Top K Principal Components (PCA)")
		pyplot.ylabel("Quadratic Weighted Kappa")
		pyplot.title("ELM - Feature Engineering with PCA")
		pyplot.savefig(png)

	@staticmethod
	def visualize_pca(x, y):
		print("PCA...")
		pca = PCA(n_components=2)
		x = pca.fit_transform(x)
		print("Post PCA, x -> " + str(x.shape))
		pyplot.scatter(x[:, 0], x[:, 1], c=y, s=9)
		pyplot.savefig("vis_pca.png")

	@staticmethod
	def visualize_tsne(x, y):
		print("T-SNE...")
		tsne_ = TSNE(n_components=2)
		x = tsne_.fit_transform(x)
		print("Post T-SNE, x -> " + str(x.shape))
		pyplot.scatter(x[:, 0], x[:, 1], c=y, s=10)
		pyplot.savefig("vis_tsne.png")

	@staticmethod
	def read_csv_(path):
		data_frame = read_csv(path)
		data = data_frame.values
		return data