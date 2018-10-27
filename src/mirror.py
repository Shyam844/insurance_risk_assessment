from matplotlib import pyplot

# CONTAINS ALL STATIC MEMBERS
class Mirror:

	@staticmethod
	def one_d(points, labels, path):
		pyplot.scatter(points, labels, c=labels, s=1)
		pyplot.savefig(path)		