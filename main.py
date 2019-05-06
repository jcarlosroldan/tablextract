from itertools import combinations
from sklearn.metrics import classification_report, confusion_matrix
from tablextract import PROPERTY_KINDS, tables
import logging

FOREVER = 10**100
NAMES = ['empty', 'data', 'header', 'indexer']
LABEL_IDS = (-1, 0, 1, 5)
LABELS = {
	'https://en.wikipedia.org/wiki/Albedo': [
		[[1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
		[[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
	],
	'https://en.wikipedia.org/wiki/Fiji': [
		[[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
		[[1, 0], [1, 0], [1, 0], [1, 0]],
		[[1, 5, 5, 5], [5, 0, 0, 0], [5, 0, 0, 0], [5, 0, 0, 0], [5, 0, 0, 0], [5, 0, 0, 0], [5, 0, 0, 0], [5, 0, 0, 0], [5, 0, 0, 0], [5, 0, 0, 0]],
		[[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
		None
	],
	'https://en.wikipedia.org/wiki/Moon': [
		[[1, 5, 5, 5], [5, 0, 0, 0], [5, -1, 0, 0]],
		[[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [-1, -1, 0, 0]],
		[[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
	]
}

def main():  # expected grid search: ~6 days, expected linear search: 1h20m
	logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(levelname)s %(relativeCreated)d %(message)s')
	logging.getLogger().addHandler(logging.StreamHandler())
	logging.info('STARTING RUN')

	for normalization in ('MinMax', 'MaxAbs', 'Standard'):
		evaluation(normalization=normalization)

	for orientation_features in ('no', 'var', 'sty-var', 'syn-var', 'sem-var', 'str-var', 'sty-var-half', 'syn-var-half', 'sem-var-half', 'str-var-half'):
		evaluation(orientation_features=orientation_features)

	for clustering_features in nonempty_subsets(PROPERTY_KINDS.keys()):
		evaluation(clustering_features=clustering_features)

	for dimensionality_reduction in ('no', 'PCA', 't-SNE', 'autoencoding', 'variability-aware'):
		evaluation(dimensionality_reduction=dimensionality_reduction)

	for clustering_method in ('k-means', 'DBSCAN'):
		evaluation(clustering_method=clustering_method)
	logging.info('ENDING RUN')

def evaluation(**kwargs):
	logging.info('TESTING with arguments %s' % kwargs)

	Y_computed = []
	Y_label = []
	for url, table_labels in LABELS.items():
		for table_label, table_computed in zip(table_labels, tables(url, request_cache_time=FOREVER, **kwargs)):
			if table_label != None:
				shp_label = len(table_label), len(table_label[0])
				shp_computed = len(table_computed.functions), len(table_computed.functions[0])
				if shp_label == shp_computed:
					Y_computed += [cell for row in table_computed.functions for cell in row]
					Y_label += [cell for row in table_label for cell in row]
				else:
					logging.warn('Ignoring table %s$%s. Label shape is %s, but computed shape is %s.' % (
						table_computed.url,
						table_computed.xpath,
						shp_label,
						shp_computed
					))

	report = classification_report(Y_label, Y_computed, labels=LABEL_IDS, target_names=NAMES)
	cm = confusion_matrix(Y_label, Y_computed, labels=LABEL_IDS)
	cm = [[' '] + NAMES] + [[name] + list(row) for name, row in zip(NAMES, cm)]
	cm = '\n'.join([' '.join(str(cell).center(8) for cell in row) for row in cm])
	logging.info('Classification report:\n%s\nConfusion matrix:\n%s' % (report, cm))

def nonempty_subsets(iterable):
	for n in range(len(iterable)):
		yield from combinations(iterable, n + 1)

if __name__ == '__main__':
	main()