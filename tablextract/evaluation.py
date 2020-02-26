from .tablextract import tables

def evaluation(normalization='MinMaxLocal', orientation_features='no', clustering_features=PROPERTY_KINDS.keys(), dimensionality_reduction='no', clustering_method='k-means'):
	pred = []
	for table, (period, orientation) in zip(dataset, knds):
		for r, row in enumerate(table['texts']):
			for c, cell in enumerate(row):
				pred.append(get_kind(period, orientation, r, c) + 1)

	print('CELL FUNCTION CLASSIFICATION REPORT')
	print(classification_report(anns, pred, labels=[1, 2], target_names=['data', 'header'], digits=4))
	print('CONFUSION MATRIX')
	print(confusion_matrix(anns, pred))

	for kind in ('horizontal listing', 'vertical listing', 'matrix'):
		kn_anns = []
		pred = []
		for table, (period, orientation) in zip(dataset, knds):
			if table['kind'] == kind:
				kn_anns.extend([cell for row in table['functions'] for cell in row])
				pred.extend([get_kind(period, orientation, r, c) + 1 for r, row in enumerate(table['functions']) for c, cell in enumerate(row)])

		print('[%s] CELL FUNCTION CLASSIFICATION REPORT' % kind)
		print(classification_report(kn_anns, pred, labels=[1, 2], target_names=['data', 'header'], digits=4))
		print('[%s] CONFUSION MATRIX' % kind)
		print(confusion_matrix(kn_anns, pred))


if __name__ == '__main__':
	params = [  # 3 x 10 x 15 x 3 x 2
		('normalization', ('MinMaxLocal', 'MinMaxGlobal', 'Standard')),
		('orientation_features', ('no', 'sty-var', 'syn-var', 'sem-var', 'str-var', 'sty-var-half', 'syn-var-half', 'sem-var-half', 'str-var-half')),
		('clustering_features', (combs for sz in range(1, 5) for combs in combinations(PROPERTY_KINDS, sz))),
		('dimensionality_reduction', ('no', 'PCA', 't-SNE')),
		('clustering_method', ('k-means', 'agglomerative'))
	]

	best_configuration = {n: v[0] for n, v in params}

	for param_name, param_values in params:
		print('Finding the best %s' % param_name)
		results = {}
		for param_value in param_values:
			print('\tTesting value %s' % param_value)
			best_configuration[param_name] = param_value
			score, report = evaluation(request_cache_time=10e10, **best_configuration)
			results[param_value] = score
		best = max(results, key=lambda k: results[k])
		best_configuration[param_name] = best
		print('\tBEST: %s' % best)

	print(best_configuration)