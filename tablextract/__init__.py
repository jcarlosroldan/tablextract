from tablextract.utils import *
from tablextract.table_processing import *

def tables(
	url,
	css_filter='table',
	xpath_filter=None,
	request_cache_time=0,
	add_image_text=True,
	add_link_urls=False,
	text_metadata_dict=None,
	normalization='min-max-global',  # min-max-global, min-max-local, standard, softmax
	clustering_features=['style', 'syntax', 'structural', 'semantic'],  # any subset of those
	dimensionality_reduction='off',  # off, pca, feature-agglomeration
	clustering_method='k-means'  # k-means, agglomerative
):
	res = []
	for table in locate(url, css_filter, xpath_filter, request_cache_time):
		try:
			segmentate(table, add_image_text, add_link_urls, url, normalization, text_metadata_dict)
			if not discriminate(table): continue
			functional_analysis(table, clustering_features, dimensionality_reduction, clustering_method)
			structural_analysis(table)
			interpret(table)
			compute_score(table)
		except:
			log_error()
			table.error = format_exc()
		res.append(table)
	return res