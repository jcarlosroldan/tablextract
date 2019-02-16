from atexit import register
from bs4 import BeautifulSoup as soup
from collections import Counter, OrderedDict
from copy import deepcopy
from datetime import datetime as dt, timedelta
try:
	from etk.extractors.date_extractor import DateExtractor
except OSError:
	from spacy.cli import download
	download('en_core_web_sm')
	from etk.extractors.date_extractor import DateExtractor
from etk.extractors.spacy_ner_extractor import SpacyNerExtractor
from math import sqrt
from numpy import array as ndarray
from nltk import download as nltk_download, pos_tag, word_tokenize
from nltk.corpus import stopwords
try:
	stopwords.words('english')
	pos_tag(word_tokenize('check'))
except:
	nltk_download('stopwords')
	nltk_download('punkt')
	nltk_download('averaged_perceptron_tagger')
from os import makedirs, remove, chmod
from os.path import dirname, abspath, exists, join
from pickle import load as pload, dump as pdump
from regex import findall, sub, compile, DOTALL, match
from requests import get
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sys import stdout, maxsize, platform
from time import strftime, sleep, time
from traceback import format_exc
from urllib.parse import urljoin
from xml.etree.cElementTree import iterparse

# --- constants ---------------------------------------------------------------

PATH_RESOURCES = join(dirname(__file__), 'resources')
PATH_LOG = join(PATH_RESOURCES, 'log_%s.txt')

PATTERN_LOG = '[%s] %s\n'
FUNCTIONS = {-1: 'empty', 0: 'data', 1: 'metadata', 2: 'context', 3: 'decorator', 4: 'total', 5: 'indexer', 6: 'factorised'}
POS_TAG_CATEGORIES = ('J', 'N', 'R', 'V', 'other')

URL_GECKODRIVER = 'https://github.com/mozilla/geckodriver/releases'

with open(join(PATH_RESOURCES, 'add_render.js'), 'r', encoding='utf-8') as fp:
	SCRIPT_ADD_RENDER = fp.read()

# --- math --------------------------------------------------------------------

def distinct(lst, uniqueness_function):
	''' Returns a list in the same order with just the elements with a distinct
	value on the uniqueness_function.
	I.e.: `distinct([1, 5, 7, 9], lambda x: x % 3)` would return [1, 5, 9].'''
	values = []
	keys = []
	for v in lst:
		k = uniqueness_function(v)
		if k not in keys:
			keys.append(k)
			values.append(v)
	return values

def dict_substract(dictionary, more_than):
	res = {}
	for k, v in dictionary.items():
		if k > more_than:
			res[k - 1] = v
		else:
			res[k] = v
	return res

def table_all_equal(table):
	if len(table) and len(table[0]):
		v = table[0][0]
		res = all(all(cell == v for cell in row) for row in table)
	else:
		res = True
	return res

# --- vector ------------------------------------------------------------------

def vectors_average(vectors):
	''' Given a list of mixed feature vectors, returns the average of all them.
	For numerical features, aritmetic average is used. For categorical ones,
	the most common is used. '''
	vectors = [v for v in vectors if len(v)]
	res = {}
	if len(vectors):
		for feat in vectors[0]:
			if type(vectors[0][feat]) == str:
				val = Counter(v[feat] for v in vectors).most_common(1)[0][0]
			else:
				val = sum(v[feat] for v in vectors) / len(vectors)
			res[feat] = val
	return res

def vectors_weighted_average(vectors):
	''' Given a list of tuples of type <weight, mixed feature vector>, returns
	the weighted average of all them. For numerical features, aritmetic average
	is used. For categorical ones, weighted frequencies are used to return the
	most common. '''
	if len(vectors) == 1: return vectors[0][1]
	res = {}
	total_weight = sum(v[0] for v in vectors)
	if total_weight == 0:
		total_weight = len(vectors)
		for n in range(total_weight):
			vectors[n][0] = 1
	vectors = [(w / total_weight, fs) for w, fs in vectors]
	for f in vectors[0][1]:
		if type(vectors[0][1][f]) == str:
			sum_feat = {}
			for weight, features in vectors:
				if features[f] in sum_feat:
					sum_feat[features[f]] += weight
				else:
					sum_feat[features[f]] = weight
			res[f] = max(sum_feat.items(), key=lambda v: v[1])[0]
		else:
			val = 0
			for weight, features in vectors:
				val += weight * features[f]
			res[f] = val
	return res

def vectors_difference(v1, v2, prefix=''):
	''' Given two mixed feature vectors, return another vector with the
	differences amongst them, taking the features of the first vector. For
	numerical features, absolute value difference is computed. For categorical
	features, Gower distance is used. '''
	res = {}
	for feat in v1:
		if type(v1[feat]) == str:
			res[prefix + feat] = 0 if v1[feat] == v2[feat] else 1
		else:
			res[prefix + feat] = abs(v1[feat] - v2[feat])
	return res

def vector_module(vector):
	''' Given a mixed feature vector, return the norm of their numerical
	attributes. '''
	nums = [v**2 for v in vector.values() if type(v) != str]
	return sqrt(sum(nums))

def binarize_categorical(vectors):
	''' Given a 2-D list of mixed feature vectors, transform every categorical
	feature into a binary one, using the seen values of all the vectors. '''
	vectors = deepcopy(vectors)
	cat_vector = next([k for k, v in cell.items() if type(v) == str] for row in vectors for cell in row if len(cell))
	for f in cat_vector:
		values = list(set(cell[f] for row in vectors for cell in row if len(cell)))
		for r, row in enumerate(vectors):
			for c, cell in enumerate(row):
				if len(cell) == 0: continue
				for v in values:
					vectors[r][c][f'{f}-{v}'] = 1 if v == cell[f] else 0
				del vectors[r][c][f]
	return vectors

# --- format ------------------------------------------------------------------

def date_stamp():
	''' Return the current timestamp. '''
	return strftime('%Y-%m-%d, %H:%M:%S')

def bytes_to_human(size, decimal_places=2):
	''' Returns a human readable file size from a number of bytes. '''
	for unit in ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']:
		if size < 1024: break
		size /= 1024
	return f'{size:.{decimal_places}f}{unit}B'

def seconds_to_human(seconds):
	''' Returns a human readable string from a number of seconds. '''
	return str(timedelta(seconds=int(seconds))).zfill(8)

# --- parsing -----------------------------------------------------------------

_find_dates_extractor = DateExtractor()
def find_dates(text):
	try:
		res = _find_dates_extractor.extract(text, prefer_language_date_order=False, detect_relative_dates=False)
		if len(res): return res[0].value
	except:
		log('info', f'ETK DateExtractor raised an error on value {text}. Using RegEx fallback instead.')

_find_entities_extractor = SpacyNerExtractor('dummy_parameter')
def find_entities(text):
	try:
		return {ext.value: ext.tag for ext in _find_entities_extractor.extract(text)}
	except:
		log('info', f'ETK SpacyNerExtractor raised an error on value {text}.')
		return {}

def lexical_densities(text, categories=POS_TAG_CATEGORIES):
	cats = [cat[0] for word, cat in pos_tag(word_tokenize(text))]
	C = len(cats)
	res = {cat: 0 for cat in categories}
	for cat in cats:
		if cat in res:
			res[cat] += 1
		else:
			res['other'] += 1
	return {k: v / C for k, v in res.items()}

# --- log ---------------------------------------------------------------------

def log(log_name, text):
	''' Logs the given text to the log specified, and prints it. '''
	text = PATTERN_LOG % (date_stamp(), text)
	print('[%s] %s' % (log_name, text), end='')
	with open(PATH_LOG % log_name, 'a', encoding='utf-8') as fp:
		fp.write(text)

def log_error():
	''' Used inside an except sentence, logs the error to the error log. '''
	log('error', format_exc())

def cache(target, args, identifier=None, cache_life=3 * 24 * 3600):
	''' Run the target function with the given args, and store it to a pickled
	cache folder using the given identifier or the name of the function. The
	next time it is executed, the cached output is returned unless cache_life
	time expires. '''
	if identifier == None: identifier = target.__name__
	identifier = sub(r'[/\\\*;\[\]\'\":=,<>]', '_', identifier)
	path = join(PATH_RESOURCES, f'.pickled/{identifier}.pk')
	makedirs(dirname(path), exist_ok=True)
	now = time()
	if exists(path):
		with open(path, 'rb') as fp:
			save_time, value = pload(fp)
		if now - save_time <= cache_life:
			return value
	res = target(*args)
	with open(path, 'wb') as fp:
		pdump((now, res), fp, protocol=3)
	return res

# --- network -----------------------------------------------------------------

def download_file(url, path=None, chunk_size=10**5):
	''' Downloads a file keeping track of the progress. '''
	if path == None: path = url.split('/')[-1]
	r = get(url, stream=True)
	total_bytes = int(r.headers.get('content-length'))
	bytes_downloaded = 0
	start = time()
	print('Downloading %s (%s)' % (url, bytes_to_human(total_bytes)))
	with open(path, 'wb') as fp:
		for chunk in r.iter_content(chunk_size=chunk_size):
			if not chunk: continue
			fp.write(chunk)
			bytes_downloaded += len(chunk)
			percent = bytes_downloaded / total_bytes
			bar = ('█' * int(percent * 32)).ljust(32)
			time_delta = time() - start
			eta = seconds_to_human((total_bytes - bytes_downloaded) * time_delta / bytes_downloaded)
			avg_speed = bytes_to_human(bytes_downloaded / time_delta).rjust(9)
			stdout.flush()
			stdout.write('\r  %6.02f%% |%s| %s/s eta %s' % (100 * percent, bar, avg_speed, eta))
	print()

_driver = None
def get_driver(headless=True, disable_images=True, open_links_same_tab=False):
	''' Returns a Firefox webdriver, and run one if there is no any active. '''
	global _driver
	if _driver == None:
		opts = Options()
		opts.set_preference('dom.ipc.plugins.enabled.libflashplayer.so', 'false')
		if open_links_same_tab:
			opts.set_preference('browser.link.open_newwindow.restriction', 0)
			opts.set_preference('browser.link.open_newwindow', 1)
		if headless: opts.set_headless()
		if disable_images: opts.set_preference('permissions.default.image', 2)
		exec_path, log_path = find_driver_path()
		try:
			_driver = Firefox(options=opts, executable_path=exec_path, log_path=log_path)
		except: # if driver not detected, try to use the default, if any
			_driver = Firefox(options=opts, log_path=log_path)
		_driver.set_page_load_timeout(15)
		register(close_driver)
	return _driver

def find_driver_path():
	null_path = '/dev/null'
	bits = 64 if maxsize > 2**32 else 32
	if platform.startswith('linux'):
		identifier = 'linux%s' % bits
	elif platform == 'win32':
		identifier = 'win%s' % bits
		null_path = 'NUL'
	elif platform == 'darwin':
		identifier = 'macos'
	else:
		log('info', 'Platform %s not identified. You will have to download and install your own webdriver from %s.' % (platform, URL_GECKODRIVER))
		return None
	driver_path = join(PATH_RESOURCES, 'geckodriver-%s' % identifier)
	if not exists(driver_path):
		page = get(URL_GECKODRIVER).text
		url_driver = urljoin(URL_GECKODRIVER, findall('href="(/mozilla/geckodriver/releases/download/.+?' + identifier + '.+?)"', page)[0])
		compressed_path = join(PATH_RESOURCES, url_driver.rsplit('/', 1)[1])
		download_file(url_driver, compressed_path)
		if compressed_path.endswith('.zip'):
			from zipfile import ZipFile
			with ZipFile(compressed_path, 'r') as zf:
				with open(driver_path, 'wb') as f:
					f.write(zf.read(zf.namelist()[0]))
		else:
			from tarfile import open as tar_open
			with tar_open(compressed_path, 'r:gz') as tf:
				with tf.extractfile('geckodriver') as gd, open(driver_path, 'wb') as f:
					f.write(gd.read())
		remove(compressed_path)
		chmod(driver_path, 755)
	return driver_path, null_path

def close_driver():
	''' Close the current Firefox webdriver, if any. '''
	global _driver
	if _driver != None:
		_driver.quit()

def get_with_render(url, render_selector='table', headless=True, disable_images=True, open_links_same_tab=False):
	''' Downloads a page and renders it to return the page source, the width,
	and the height in pixels. Elements on the subtree selected using
	render_selector contain a data-computed-style attribute and a data-xpath. '''
	driver = get_driver(headless, disable_images, open_links_same_tab)
	driver.get(url)
	driver.execute_script(SCRIPT_ADD_RENDER, render_selector)
	sleep(.5)
	return driver.page_source