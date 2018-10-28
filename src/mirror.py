from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# CONTAINS ALL STATIC MEMBERS
class Mirror:

	@staticmethod
	def one_d(points, labels, path):
		pyplot.scatter(points, labels, c=labels, s=1)
		pyplot.savefig(path)		

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