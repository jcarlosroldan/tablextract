from tablextract.utils import *

PADDING_CELL = soup('<td data-padding-cell></td>', 'html.parser').td
FIND_DIGITS = compile(r"\d+").findall
STOPWORDS = stopwords.words("english")
FIND_STOPWORDS = lambda txt: [w for w in findall(r"[^\s]+", txt.lower()) if w in STOPWORDS]
COLOR_STYLE_PROPERTIES = ["color", "background-color", "border-left-color", "border-bottom-color", "border-right-color", "border-top-color", "outline-color"]
CATEGORICAL_STYLE_PROPERTIES = ["text-align", "font-family", "text-transform", "text-decoration", "vertical-align", "display", "tag"]
NUMERIC_STYLE_PROPERTIES = {"border-bottom-width": (-40, 40), "border-left-width": (-40, 40), "border-right-width": (-40, 40), "border-top-width": (-40, 40), "font-size": (0, 60), "padding-bottom": (-40, 40), "padding-left": (-40, 40), "padding-right": (-40, 40), "padding-top": (-40, 40), "font-weight": (400, 700)}
NUMERIC_STYLE_PROPERTIES = {k: (mn, mx - mn) for k, (mn, mx) in NUMERIC_STYLE_PROPERTIES.items()}
COMPUTED_STYLE_PROPERTIES = ["tag", "relative-width", "relative-height"]
DENSITY_SYNTAX_PROPERTIES = {"lowercase": r"\p{Ll}", "uppercase": r"\p{Lu}", "alphanumeric": r"\w", "digit": r"\d", "whitespace": r"\s", "symbol": r"[^\w\s]", "token": r"[^\s]+"}
DENSITY_SYNTAX_PROPERTIES = {f'density-{k}': compile(v).findall for k, v in DENSITY_SYNTAX_PROPERTIES.items()}
DENSITY_SYNTAX_PROPERTIES["density-stopwords"] = FIND_STOPWORDS
BOOLEAN_SYNTAX_PROPERTIES = {"capitalised": r"(\p{Lu}(\p{Ll}+|\.)\s)*\p{Lu}(\p{Ll}+|\.)", "allcaps": r"\p{Lu}+", "money": r"[\$£]\s*\d+([.,] ?\d+)?\s*$|\d+([.,] ?\d+)?\s*€", "amount": r"([\-\+]\s*)?[\d ,\.]+", "range": r"((\+\-)?\s*\d[\d(,\s?)\.]*[\-\–(to),\s]*)+", "empty": r""}
BOOLEAN_SYNTAX_PROPERTIES = {f'match-{k}': compile("^%s$" % v).match for k, v in BOOLEAN_SYNTAX_PROPERTIES.items()}
BOOLEAN_SYNTAX_PROPERTIES["match-date"] = find_dates
BOOLEAN_SYNTAX_PROPERTIES["match-location"] = lambda x: len([it for it in find_entities(x).items() if it[1] == 'GPE']) > 0
BOOLEAN_SYNTAX_PROPERTIES["match-person"] = lambda x: len([it for it in find_entities(x).items() if it[1] == 'PERSON']) > 0
PROPERTY_KINDS = {
	'style': ['background-color-b', 'background-color-g', 'background-color-r', 'border-bottom-color-b', 'border-bottom-color-g', 'border-bottom-color-r', 'border-bottom-width', 'border-left-color-b', 'border-left-color-g', 'border-left-color-r', 'border-left-width', 'border-right-color-b', 'border-right-color-g', 'border-right-color-r', 'border-right-width', 'border-top-color-b', 'border-top-color-g', 'border-top-color-r', 'border-top-width', 'color-b', 'color-g', 'color-r', 'display', 'font-family', 'font-size', 'font-weight', 'outline-color-b', 'outline-color-g', 'outline-color-r', 'padding-bottom', 'padding-left', 'padding-right', 'padding-top', 'text-align', 'text-decoration', 'text-transform', 'vertical-align'],
	'syntax': ['density-alphanumeric', 'density-digit', 'density-lowercase', 'density-stopwords', 'density-symbol', 'density-token', 'density-uppercase', 'density-whitespace', 'match-allcaps', 'match-amount', 'match-capitalised', 'match-date', 'match-empty', 'match-location', 'match-money', 'match-person', 'match-range'],
	'structural': ['children', 'colspan', 'rowspan', 'tag']
}
PROPERTY_KINDS = {k: v + ['%s-variability-%s' % (dim, feat) for feat in v for dim in ['row', 'col', 'tab']] for k, v in PROPERTY_KINDS.items()}
MAX_SPAN = 150

def locate(url, document):
	res = []
	if document.select_one('.noarticletext') != None:
		log('info', f'Article {url} does not exist.')
	else:
		for table in document.select('table[data-xpath]'):
			child_tables = len(table.select('table'))
			rows = len(table.select('tr'))
			cols = len(table.select('td')) / rows if rows > 0 else 0
			if rows > 1 and cols > 1 and not child_tables:
				res.append(Table(url, table['data-xpath'], table))
	return res

def segmentate(table):
	elements = [[cell for cell in row.select('th,td')] for row in table.element.select('tr')]
	elements, context = clean_table(elements)
	features = []
	texts = []
	for row in elements:
		row_data = []
		row_text = []
		for cell in row:
			if 'data-padding-cell' in cell.attrs:
				cell_feats = {}
			else:
				cell_feats = extract_features(cell, len(elements), len(elements[0]))
			row_data.append(cell_feats)
			row_text.append(extract_text(cell))
		features.append(row_data)
		texts.append(row_text)
	table.elements = elements
	table.features = features
	table.texts = texts
	table.context = context
	if len(table.elements) and len(table.elements[0]):
		place_context(table)
		add_variability(table)

def clean_table(table):
	# convert colspans to int and avoid huge span
	for r, row in enumerate(table):
		for c, cell in enumerate(row):
			if 'colspan' in cell.attrs:
				try:
					cell['colspan'] = min(MAX_SPAN, int(cell['colspan']))
				except:
					cell['colspan'] = 1
			if 'rowspan' in cell.attrs:
				try:
					cell['rowspan'] = min(MAX_SPAN, int(cell['rowspan']))
				except:
					cell['rowspan'] = 1
	# split and copy colspan
	spanned_cells = []
	for r, row in enumerate(table):
		for c, cell in enumerate(row):
			if 'colspan' in cell.attrs and (r, c) not in spanned_cells:
				for n in range(cell['colspan'] - 1):
					table[r].insert(c, cell)
					spanned_cells.append((r, c + 1 + n))
	# split and copy rowspan
	spanned_cells = []
	for c in range(len(table[0])):
		for r in range(len(table)):
			if c < len(table[r]) and 'rowspan' in table[r][c].attrs and (r, c) not in spanned_cells:
				for n in range(1, min(len(table) - r, table[r][c]['rowspan'])):
					table[r + n].insert(c, table[r][c])
					spanned_cells.append((r + n, c))
	# pad with trailing cells
	width = max(len(row) for row in table)
	table = [row + [PADDING_CELL] * (width - len(row)) for row in table]
	# transform empty cells into padding cells
	# XXX extraer aquí los texts e imágenes
	for r, row in enumerate(table):
		for c, cell in enumerate(row):
			if not len(' '.join(t.strip() for t in cell.find_all(text=True)).strip()):
				table[r][c] = PADDING_CELL
	# remove non-data rows until no changes are applied to the table
	changed = True
	context_rows = {}
	context_cols = {}
	while changed:
		changed = False
		# rows removal
		seen_before = []
		r = 0
		while r < len(table):
			row = table[r]
			row_text = [cell.text.strip() for cell in row]
			all_empty = not any(len(cell) for cell in row_text)
			all_padding = all('rowspan' in cell.attrs and cell['rowspan'] > 1 or cell == PADDING_CELL for cell in row)
			is_repeated = row_text in seen_before
			seen_before.append(row_text)
			full_colspan = len(row) and all(row[cell - 1] == row[cell] for cell in range(1, len(row)))
			if all_empty or all_padding or is_repeated or full_colspan:
				if full_colspan and len(row_text[0]):
					if r in context_rows:
						context_rows[r].append(row[0])
					else:
						context_rows[r] = [row[0]]
				table = table[:r] + table[r + 1:]
				context_rows = dict_substract(context_rows, r)
				changed = True
			else:
				r += 1
		# cols removal
		if not len(table): break
		seen_before = []
		c = 0
		while c < len(table[0]):
			col = [row[c] for row in table]
			col_text = [cell.text.strip() for cell in col]
			all_empty = not any(len(cell) for cell in row_text)
			all_padding = all('colspan' in cell.attrs and cell['colspan'] > 1 or cell == PADDING_CELL for cell in col)
			is_repeated = col_text in seen_before
			seen_before.append(col_text)
			full_rowspan = len(col) and all(col[cell - 1] == col[cell] for cell in range(1, len(col)))
			if all_empty or all_padding or is_repeated or full_rowspan:
				if full_rowspan and len(col_text[0]):
					if c in context_cols:
						context_cols[c].append(col[0])
					else:
						context_cols[c] = [col[0]]
				table = [row[:c] + row[c + 1:] for row in table]
				context_cols = dict_substract(context_cols, c)
				changed = True
			else:
				c += 1
	context = {}
	for k, v in context_rows.items():
		for nv, vv in enumerate(v):
			context[('r', k, nv)] = vv
	for k, v in context_cols.items():
		for nv, vv in enumerate(v):
			context[('c', k, nv)] = vv
	return table, context

def extract_features(element, rows, cols):
	if 'data-computed-style' not in element.attrs:
		return None
	css_properties = [prop.split(':') for prop in element['data-computed-style'].split('|')]
	css_properties = {k: v for k, v in css_properties}
	# compute style properties
	res = {}
	for p in COLOR_STYLE_PROPERTIES:
		val = css_properties[p]
		res[p + '-r'], res[p + '-g'], res[p + '-b'] = [float(c) / 255 for c in FIND_DIGITS(val)[:3]]
	for p in CATEGORICAL_STYLE_PROPERTIES:
		if p != 'tag':
			res[p] = css_properties[p]
	for p, (mn, wide) in NUMERIC_STYLE_PROPERTIES.items():
		val = css_properties[p]
		res[p] = max(0, min(1, (float(FIND_DIGITS(val)[0]) - mn) / wide))
	res['tag'] = element.name
	res['width'] = max(0, min(1, float(css_properties['width'])))
	res['height'] = max(0, min(1, float(css_properties['height'])))
	# compute syntax properties
	text = extract_text(element, add_image_text=False)
	ln = len(text)
	for p, reg in DENSITY_SYNTAX_PROPERTIES.items():
		if ln:
			res[p] = len(reg(text)) / ln
		else:
			res[p] = 0
	for p, reg in BOOLEAN_SYNTAX_PROPERTIES.items():
		res[p] = 1 if reg(text) else 0
	# add children render info
	area = res['width'] * res['height']
	nodes = []
	for c in element.find_all(recursive=False):
		if len(c.text.strip()) and 'data-computed-style' in c.attrs:
			child = extract_features(c, rows, cols)
			if child == None:
				continue
			child_area = child['width'] * child['height']
			area -= child_area
			nodes.append([child_area, child])
	nodes.insert(0, [area, res])
	res = vectors_weighted_average(nodes)
	res['tag'] = element.name
	res['children'] = max(0, min(1, len(element.find_all()) / 5))
	if res['tag'] in ['th', 'td']:
		res['colspan'] = max(0, min(1, int(css_properties['colspan']) / cols))
		res['rowspan'] = max(0, min(1, int(css_properties['rowspan']) / rows))
		del res['width']
		del res['height']
	return res

def extract_text(element, add_image_text=True):
	res = []
	for desc in element.descendants:
		if desc.name == None:
			res.append(desc)
		elif add_image_text and desc.name == 'img':
			if desc.has_attr('alt') and len(desc['alt']):
				res.append('(%s)' % desc['alt'])
			elif desc.has_attr('title') and len(desc['title']):
				res.append('(%s)' % desc['title'])
			elif desc.has_attr('src') and len(desc['src']):
				res.append('(%s)' % desc['src'].rsplit('/')[-1].split('.')[0])
			else:
				print(desc)
	return ' '.join([r.strip() for r in res]).strip()

def place_context(table):
	rows, cols = table.rows(), table.cols()
	context_rows = list(sorted(
		(k[1], k[2], extract_features(v, rows, cols), v)
		for k, v in table.context.items() if k[0] == 'r')
	)
	top_rows = [row for row in context_rows if row[0] == 0]
	middle_rows = [row for row in context_rows if 0 < row[0] < table.rows() and row[1] == 0]
	bottom_rows = [row for row in context_rows if row[0] == table.rows()]
	if len(middle_rows):
		middle_average = vectors_average([r[2] for r in middle_rows])
		diff_top_row = vector_module(vectors_difference(middle_average, top_rows[-1][2])) if len(top_rows) else 2
		diff_bot_row = vector_module(vectors_difference(middle_average, bottom_rows[0][2])) if len(bottom_rows) else 2
		if diff_bot_row < diff_top_row and diff_bot_row < 2:
			if diff_bot_row < 2:
				folded = middle_rows + [bottom_rows[0]]
			else:
				folded = middle_rows
			folded = {r: (feats, elem, extract_text(elem), r_i) for r, r_i, feats, elem in folded}
			for r in range(table.rows()):
				if r + 1 in folded:
					table.features[r].append(folded[r + 1][0])
					table.elements[r].append(folded[r + 1][1])
					table.texts[r].append(folded[r + 1][2])
					del table.context[('r', r + 1, folded[r + 1][3])]
				else:
					table.features[r].append({})
					table.elements[r].append(PADDING_CELL)
					table.texts[r].append('')
		else:
			if diff_top_row < 2:
				factorised = [top_rows[-1]] + middle_rows
			else:
				factorised = middle_rows
			factorised = {r: (feats, elem, extract_text(elem), r_i) for r, r_i, feats, elem in factorised}
			value = None
			for r in range(table.rows()):
				if r in factorised:
					table.features[r].append(factorised[r][0])
					table.elements[r].append(factorised[r][1])
					table.texts[r].append(factorised[r][2])
					del table.context[('r', r, factorised[r][3])]
				else:
					table.features[r].append({})
					table.elements[r].append(PADDING_CELL)
					table.texts[r].append('')
	table.context = {'_'.join(map(str, k)): extract_text(v) for k, v in table.context.items()}

def add_variability(table):
	if len(table.features) == 0:
		table.variabilities = {'row': 0, 'col': 0, 'table': 0}
		return
	xpath_getter = lambda x: x[1]['data-xpath'] if 'data-xpath' in x[1].attrs else ''
	row_features = [
		vectors_average([c_ft for c_ft, c_el in distinct(zip(r_ft, r_el), xpath_getter)])
		for r_ft, r_el in zip(table.features, table.elements)
	]
	col_features = [
		vectors_average(c_ft for c_ft, c_el in distinct(
			[(r_ft[c], r_el[c]) for r_ft, r_el in zip(table.features, table.elements)],
			xpath_getter
		))
		for c in range(table.cols())
	]
	tab_features = vectors_average(
		c_ft for c_ft, c_el in distinct(
			[
				(c_ft, c_el)
				for r_ft, r_el in zip(table.features, table.elements)
				for c_ft, c_el in zip(r_ft, r_el)
			],
			xpath_getter
		)
	)

	total_row_variability = []
	total_col_variability = []
	total_table_variability = []
	total_style_row_variability = []
	total_style_col_variability = []
	total_style_table_variability = []
	total_syntax_row_variability = []
	total_syntax_col_variability = []
	total_syntax_table_variability = []

	for r, row in enumerate(table.features):
		for c, cell in enumerate(row):
			if len(cell) == 0: continue
			row_variabilities = vectors_difference(cell, row_features[r], prefix='row-variability-')
			col_variabilities = vectors_difference(cell, col_features[c], prefix='col-variability-')
			table_variabilities = vectors_difference(cell, tab_features, prefix='tab-variability-')
			table.features[r][c] = {**cell, **row_variabilities, **col_variabilities, **table_variabilities}
			total_row_variability.append(vector_module(row_variabilities))
			total_col_variability.append(vector_module(col_variabilities))
			total_table_variability.append(vector_module(table_variabilities))
			total_style_row_variability.append(vector_module({k: v for k, v in row_variabilities.items() if k in PROPERTY_KINDS['style']}))
			total_style_col_variability.append(vector_module({k: v for k, v in col_variabilities.items() if k in PROPERTY_KINDS['style']}))
			total_style_table_variability.append(vector_module({k: v for k, v in table_variabilities.items() if k in PROPERTY_KINDS['style']}))
			total_syntax_row_variability.append(vector_module({k: v for k, v in row_variabilities.items() if k in PROPERTY_KINDS['syntax']}))
			total_syntax_col_variability.append(vector_module({k: v for k, v in col_variabilities.items() if k in PROPERTY_KINDS['syntax']}))
			total_syntax_table_variability.append(vector_module({k: v for k, v in table_variabilities.items() if k in PROPERTY_KINDS['syntax']}))

	table.variabilities = {
		'row': sum(total_row_variability) / len(total_row_variability),
		'col': sum(total_col_variability) / len(total_col_variability),
		'table': sum(total_table_variability) / len(total_table_variability),
		'style-row': sum(total_style_row_variability) / len(total_style_row_variability),
		'style-col': sum(total_style_col_variability) / len(total_style_col_variability),
		'style-table': sum(total_style_table_variability) / len(total_style_table_variability),
		'syntax-row': sum(total_syntax_row_variability) / len(total_syntax_row_variability),
		'syntax-col': sum(total_syntax_col_variability) / len(total_syntax_col_variability),
		'syntax-table': sum(total_syntax_table_variability) / len(total_syntax_table_variability)
	}

def functional_analysis(table):
	# ensure variables are normalised
	for row in table.features:
		for cell in row:
			for k, v in cell.items():
				if not (type(v) == str or -1e-12 <= v <= 1 + 1e-12):
					log('error', f'Feature {k} outside the boundaries: {v}.')
	xpath_table = [[cell['data-xpath'] if cell != PADDING_CELL else '' for cell in row] for row in table.elements]
	kind_functions = {}
	for kind, feature_names in PROPERTY_KINDS.items():
		vector_table = []
		for row in table.features:
			vector_table.append([])
			for cell in row:
				vector_table[-1].append({})
				for k, v in cell.items():
					if k in feature_names:
						vector_table[-1][-1][k] = v
		kind_functions[kind] = cluster_vector_table(vector_table, xpath_table)

	functions = [[-1] * table.cols() for _ in range(table.rows())]
	for r in range(table.rows()):
		for c in range(table.cols()):
			if len(table.features[r][c]):
				cell_function = {0: [], 1: []}
				for kind in PROPERTY_KINDS:
					cell_function[kind_functions[kind][0][r][c]].append(kind_functions[kind][1])
				cell_function = max((sum(v) / len(v), k) for k, v in cell_function.items() if len(v))[1]
				functions[r][c] = cell_function

	table.functions = functions

def cluster_vector_table(vector_table, xpath_table):
	# get cluster values
	rows, cols = len(xpath_table), len(xpath_table[0])
	features = binarize_categorical(vector_table)
	cells = {
		xp: [f[1] for f in sorted(ft.items())]
		for r_ft, r_xp in zip(features, xpath_table)
		for ft, xp in zip(r_ft, r_xp)
		if len(ft)
	}
	cells = list(cells.items())
	km = KMeans(n_clusters=2).fit([c[1] for c in cells])
	cells = {xpath: int(lab) for (xpath, _), lab in zip(cells, km.labels_)}
	functions = []
	for r in range(rows):
		functions.append([])
		for c in range(cols):
			if len(features[r][c]):
				functions[-1].append(cells[xpath_table[r][c]])
			else:
				functions[-1].append(-1)
	if table_all_equal(functions):
		return functions, 1
	# estimate which cluster should be the headers
	header_likelyhood = {0: [], 1: []}
	# estimator 1: occurrences in first row or col
	for c in [0, 1]:
		amount = len([1 for cell in functions[0] if cell == c]) / cols
		amount += len([1 for row in functions if row[0] == c]) / rows
		header_likelyhood[c].append(amount / 2)
	# estimator 2: amount of cells
	for c in [0, 1]:
		amount = len([1 for row in functions for cell in row if cell == c])
		header_likelyhood[c].append(1 - amount / (rows * cols))
	# estimator 3: closeness to first cell
	pos = [[], []]
	for r, row in enumerate(functions):
		for c, cell in enumerate(row):
			pos[cell].append((r, c))
	pos = [[sum([e[n] for e in elem]) / len(elem) for n in range(2)] for elem in pos]
	pos = [sqrt((x / rows)**2 + (y / cols)**2) for x, y in pos]
	for c, d in enumerate(pos):
		header_likelyhood[c].append(1 - d)
	# combine estimators
	for k in header_likelyhood:
		header_likelyhood[k] = sum(header_likelyhood[k]) / len(header_likelyhood[k])
	exchange = header_likelyhood[0] > header_likelyhood[1]
	# apply result
	if exchange:
		exchange_dict = {-1: -1, 0: 1, 1: 0}
		functions = [[exchange_dict[c] for c in row] for row in functions]
	confidence = abs(header_likelyhood[0] - header_likelyhood[1])
	# XXX class correction
	return functions, confidence

def structural_analysis(table):
	# TODO double headers
	merge_headers(table)
	# TODO split headers
	rv, cv, tv = table.variabilities['row'], table.variabilities['col'], table.variabilities['table']
	all_headers = all(c == 1 for row in table.functions for c in row if c != -1)
	all_data = all(c == 0 for row in table.functions for c in row if c != -1)
	first_row_header = all(c == 1 for c in table.functions[0] if c != -1)
	first_col_header = all(row[0] == 1 for row in table.functions if row[0] != -1)
	if all_headers or all_data:
		min_variability = min(rv, cv, tv)
		if min_variability == rv:
			table.kind = 'horizontal listing'
			if all_headers:
				table.functions = [[1] * table.cols()] + [[0] * table.cols()] * (table.rows() - 1)
		elif min_variability == cv:
			table.kind = 'vertical listing'
			if all_headers:
				table.functions = [[[1] + [0] * table.cols()] for _ in range(table.rows())]
		else:
			table.kind = 'enumeration'
	elif first_row_header and first_col_header:
		table.kind = 'matrix'
		for c in range(1, table.cols()): table.functions[0][c] = 5
		for r in range(1, table.rows()): table.functions[r][0] = 5
	elif first_row_header:
		table.kind = 'horizontal listing'
	elif first_col_header:
		table.kind = 'vertical listing'

def merge_headers(table):
	# TODO merge hierarchical headers
	pass

def interpret(table):
	if table.kind == 'enumeration' or table.kind == 'unknown':
		res = [{'attribute_0': text} for row in table.texts for text in row if len(text)]
	elif table.kind == "matrix":
		res = [
			{table.texts[0][0]: table.texts[r][0], 'attribute_0': table.texts[0][c], 'attribute_1': table.texts[r][c]}
			for r, row in enumerate(table.functions) for c, cell in enumerate(row) if cell == 0
		]
	elif table.kind == 'horizontal listing':
		res = [{k: v for k, v in zip(table.texts[0], row)} for row in table.texts[1:]]
	elif table.kind == 'vertical listing':
		first_column = [row[0] for row in table.texts]
		res = [{k: v for k, v in zip(first_column, [r[c] for r in table.texts])} for c in range(1, len(table.functions[0]))]
	table.record = res

_compute_score_functions = {0: 1, -1: 0.5, 1: 0}
def compute_score(table):
	if table.kind == 'unknown':
		table.score = 0
	else:
		variability_score = 1
		max_var = max(table.variabilities.values())
		if table.kind == 'horizontal listing':
			header_area = table.features[0]
			data_area = [cell for row in table.functions[1:] for cell in row]
			if max_var != table.variabilities['row']: variability_score = .75
		elif table.kind == 'vertical listing':
			header_area = [row[0] for row in table.features]
			data_area = [cell for row in table.functions for cell in row[1:]]
			if max_var != table.variabilities['col']: variability_score = .75
		elif table.kind == 'matrix':
			header_area = table.features[0] + [row[0] for row in table.features]
			data_area = [cell for row in table.functions[1:] for cell in row[1:]]
			if max_var != table.variabilities['table']: variability_score = .75
		elif table.kind == 'enumeration':
			header_area = []
			data_area = [cell for row in table.functions for cell in row]
			if max_var != table.variabilities['table']: variability_score = .75
		if len(header_area):
			numeric_header_score = [cell['density-digit'] for cell in header_area if len(cell)]
			numeric_header_score = 1 - sum(numeric_header_score) / len(numeric_header_score)
		else:
			numeric_header_score = 1
		if len(data_area):
			data_header_score = sum(c != 1 for c in data_area) / len(data_area)
		else:
			data_header_score = 1
		element = soup('<table>%s</table>' % ''.join(['<tr>%s</tr>' % ''.join(map(str, row)) for row in table.elements]), 'html.parser').table
		tokens = [kw.strip() for text in element.find_all(text=True, recursive=True) for kw in text.split(' ')]
		tokens_after = [kw.strip() for rec in table.record for k, v in rec.items() for kw in k.split(' ') + v.split(' ')]
		information_loss_score = len([t for t in tokens if t in tokens_after]) / len(tokens)
		table.score = (variability_score * numeric_header_score * data_header_score * information_loss_score) ** (1 / 4)

class Table:
	def __init__(self, url=None, xpath=None, element=None):
		self.url = url
		self.xpath = xpath
		self.element = element
		self.elements = None
		self.features = None
		self.texts = None
		self.context = None
		self.functions = None
		self.variabilities = {'row': None, 'col': None, 'table': None}
		self.kind = 'unknown'
		self.record = None
		self.score = None
		self.error = None

	def rows(self):
		return len(self.elements)

	def cols(self):
		if self.rows():
			return len(self.elements[0])
		else:
			return 0

	def cells(self):
		if len(self.features):
			return len(self.features) * len(self.features[0])
		else:
			return 0

	def __str__(self):
		res = 'Table(url=%s, xpath=%s' % (self.url, self.xpath)
		if self.error:
			res += ', error=%s)' % self.error.strip().rsplit('\n', 1)[1]
		else:
			res += ')'
		return res