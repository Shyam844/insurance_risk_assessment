from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pandas import read_csv
from seaborn import heatmap

# CONTAINS ALL STATIC MEMBERS
class Mirror:

	@staticmethod
	def plot_heatmap(data, filename):
		axis = heatmap(data, vmin=0, vmax=1)
		axis.set_xlabel("Features")
		axis.set_ylabel("Samples")
		axis.set_title("Train_x Heatmap")
		fig = axis.get_figure()
		fig.savefig(filename)

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