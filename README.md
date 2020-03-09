# Tablextract

This Python 3 library extracts the information represented in any HTML table. This project has been developed in the context of the paper `TOMATE: On extracting information from HTML tables`.

Some of the main features of this library are:

* Context location: Context information is detected, both inside and outside the table.
* Cell role detection: Classification of cells in headers and data based on the style, syntax, structure and semantics.
* Layout detection: Automatic identification of horizontal listings, vertical listings, matrices and enumerations.
* Record extraction: Identified tables are extracted as a list of dictionaries, each one being a database record.

Some features that will be added soon:

* Cell correction: Analysis of the orientation of the table to fix wrongly labelled cells.
* Totals detection: Detect totalling cells automatically from the data.

## How to install

You can install this library via pip using:
```pip install tablextract```

## Usage example

```python
>>> from pprint import pprint
>>> from tablextract import tables
>>>
>>> ts = tables('https://en.wikipedia.org/wiki/Fiji')
>>> ts
[
    Table(url=https://en.wikipedia.org/wiki/Fiji, xpath=.../div[4]/div[1]/table[2]),
    Table(url=https://en.wikipedia.org/wiki/Fiji, xpath=.../div[4]/div[1]/table[3]),
    Table(url=https://en.wikipedia.org/wiki/Fiji, xpath=.../div[4]/div[1]/table[4])
]
>>> ts[0].record
[
    {'Confederacy': 'Burebasaga', 'Chief': 'Ro Teimumu Vuikaba Kepa'},
    {'Confederacy': 'Kubuna', 'Chief': 'Vacant'},
    {'Confederacy': 'Tovata', 'Chief': 'Ratu Naiqama Tawake Lalabalavu'}
]
>>> ts[2].record  # it automatically identifies that it's laid out vertically
[
    {
        'English': 'Hello/hi',
        'Fijian': 'bula',
        'Fiji Hindi': 'नमस्ते (namaste)'
    }, {
        'English': 'Good morning',
        'Fijian': 'yadra (Pronounced Yandra)',
        'Fiji Hindi': 'सुप्रभात (suprabhat)'
    }, {
        'English': 'Goodbye',
        'Fijian': 'moce (Pronounced Mothe)',
        'Fiji Hindi': 'अलविदा (alavidā)'
    }
]
```

This library only have one function `tables`, that returns a list of `Table` objects.

`tables(url, css_filter='table', xpath_filter=None, request_cache_time=None, add_link_urls=False, normalization'min-max-global', clustering_features=['style', 'syntax', 'structural', 'semantic'], dimensionality_reduction='off', clustering_method='k-means')`

* `url: str`: URL of the site where tables should be downloaded from.
* `css_filter: str`: Return just tables that match the CSS selector.
* `xpath_filter: str`: Return just tables that match the XPath.
* `request_cache_time: int`: Cache the downloaded documents for that number of seconds.
* `add_image_text: bool`: Extract the image title/alt/URL as part of the cell text in `Table.texts`.
* `add_link_urls: bool`: Extract the links URL as part of the cell text in `Table.texts`.
* `text_metadata_dict: dict`: Dictionary of cell texts and likelyhood of it being meta-data. See [Meta-data probability corpus](https://github.com/juancroldan/tablextract/releases/tag/1.4.1).
* `normalization: str`: The kind of normalization applied to the features. Allowed values are `min-max-global` to use MinMax normalization with values obtained from a big corpus of tables after removing outliers, `min-max-local` to use MinMax normalization with the minimum and maximum values of each feature in the table, `standard` to apply a Standard normalization and `softmax` to apply a SoftMax normalization.
* `clustering_features: list`: The clustering feature groups that are used to identify the cell functions. Any non-empty subset of style', 'syntax', 'structural' and 'semantic' is allowed.
* `dimensionality_reduction`: The technique used to reduce the cells dimensionality before clustering. Allowed values are `off` to disable it, `pca` and `feature-agglomeration`.
* `clustering_method`: The method used to cluster the cells. Allowed methods are `k-means` and `agglomerative`.

Each `Table` object has the following properties and methods:

* `cols(): int`: Number of columns of the table.
* `rows(): int`: Number of rows of the table.
* `cells(): int`: Number of cells of the table (same as `table.cols() * table.rows()`).

* `error: str or None`: If an error has occurred during table extraction, it contains the stacktrace of it. Otherwise, it is None.
* `url: str`: URL of the page from where the table was extracted.
* `xpath: str`: XPath of the table within the page.
* `element: bs4.element.Tag`: BeautifulSoup element that represents the table.
* `elements: list of list of bs4.element.Tag`: 2D table of BeautifulSoup elements that represents the table after cell segmentation.
* `texts: list of list of str`: 2D table of strings that represents the text of each cell.
* `context: dict of {tuple, str}`: Texts inside or outside the table that provides contextual information for it. The keys of the dictionary represents the context position.
* `features: list of list of dict of {str, float/str}`: 2D table of feature vectors for each cell in the table.
* `functions: list of list of int`: 2D table of functions of the cells of the table. Functions can be EMPTY (-1), DATA (0), or METADATA(1).
* `kind: str`: Type of table extracted. Types can be 'horizontal listing', 'vertical listing', 'matrix', 'enumeration' or 'unknown'.
* `record: list of dict of {str, str}`: Database-like records extracted from the table.
* `score: float`: Estimation of how properly the table was extracted, between 0 and 1, being 1 a perfect extraction.

## Notes

If you update this library and you get the error `sre_constants.error: bad escape \p at position 257`, you might be using a corrupted environment. You can either:

* Try to fix your current environment by forcing the download of SpaCy models: `python3 -m spacy download en`
* Create a new environment to work with: `python3 -m venv my_new_env`, `source my_new_env/bin/activate`

## Changes

### v1.5

Released on Mar 03, 2020.

* Added parameter text_metadata_dict to `tables`.
* Empty table bugfixes to introduce individual step processing.

### v1.4

Released on Feb 26, 2020.

* Added parameter normalization to `tables`.
* Added parameter clustering_features to `tables`.
* Added parameter dimensionality_reduction to `tables`.
* Added parameter clustering_method to `tables`.
* Bugfix: When averaging feature values of cell subtrees, some features were outside the bounds due to child nodes having larger area than parent nodes.
* Bugfix: Dependency regex not installed at setup.
* Simplified main call to `tables`.
* Overall speed and efficiency increase.

### v1.3

Released on May 12, 2019.

* Added parameter add_link_urls to `tables`.
* Improved algorithm for context placement.
* Text is now extractd only while cleaning the table, making the parsing faster.
* Improved empty cell detection.
* Empty cells are now preserved both at orientation correction and matrix correction.
* Lexical density is now not computed for empty cells.
* Improved format for `render_tabular_array`.
* Hotfix: 2xn or nx2 tables with the cell (0, 1) or (1, 0) empty could not compute the orientation correction.
* Bugfix: Some tables with less than 3 tds (but some ths) were not extracted.
* Bugfix: when images were added to the cell text, that text was used when computing the textual features.
* Bugfix: Some fullspan tables were not properly extracted.
* Bugfix: Feature weighting was wrong when the child node was bigger than the parent node.

### v1.2

Released on Mar 25, 2019.

* Named entity detection is not performed during feature extraction stage.
* Removed Wikipedia-specific selector constraint.
* The previous and next non-inline tags with text relative to the table is extracted as context.
* The hierarchy of header tags h1-h6 is extracted as context.
* More tables are extracted on the location stage.
* Repeated headers and hierarchic headers are more clear.
* Added parameter to keep the links URL at the text.
* *Bugfix* Tables with last row empty were not extracted.
* *Bugfix* The Table.record was not extracted for some vertical listings.

### v1.1

Released on Feb 05, 2019.

* Orientation is automatically detected to fix some table cell functions.
* New features are extracted from the cells: POS tagging densities, relative column and row indices, first-char-type and last-char-type.
* Hierarchical, factorised, and some periodical headers are segmented properly before the extraction.
* Instead of discarding tables with tables inside and then discarding tables smaller than 2x2, it first removes the small tables and then discards tables with tables inside, in order to get more results.
* Texts and images are extracted before discarding repeated cells, to avoid discarding rows with changing images.
* Cache is disabled by default.
* Readme documentation improved.

### v1.0

Released on Jan 24, 2019.

* Before using Selenium, geckodriver is automatically downloaded for Linux, Windows and Mac OS.
* The Firefox process is closed automatically when the process ends.
* Geckodriver `quit` is called instead of `close`.
* Side-projects has been moved from this core project to tablextract-server and datamart.
* Fixed project imports and setup.
* More readable Table objects.

### v0.0.

Released on Jan 22, 2019.

* Initial package upload.
* Removed side projects to tablextractserver and datamart.