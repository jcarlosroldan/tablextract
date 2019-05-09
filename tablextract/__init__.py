from .utils import *
from .table_processing import *

def tables(
	url,
	css_filter='table',
	xpath_filter=None,
	request_cache_time=0,
	add_link_urls=False,
	normalization='MinMax',  # MinMax, MaxAbs, Standard
	orientation_features='no',  # var, sty-var, syn-var, sem-var, str-var, sty-var-half, syn-var-half, sem-var-half
	clustering_features=PROPERTY_KINDS.keys(),  # any non-empty subset of this
	dimensionality_reduction='no',  # PCA, t-SNE, autoencoding, variability aware
	clustering_method='k-means'  # k-means, DBSCAN
):
	document = cache(get_with_render, (url, css_filter), identifier=url, cache_life=request_cache_time)
	document = soup(document, 'html.parser')
	res = []
	for table in locate(url, document):
		if xpath_filter == None or table.xpath == xpath_filter:
			try:
				segmentate(table, add_link_urls, normalization)
				if len(table.features) < 2 or len(table.features[0]) < 2: continue
				functional_analysis(table, orientation_features, clustering_features, dimensionality_reduction, clustering_method)
				structural_analysis(table)
				interpret(table)
				compute_score(table)
			except:
				log_error()
				table.error = format_exc()
			res.append(table)
	return res