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

`tables(url, css_filter='table', xpath_filter=None, request_cache_time=None)`

* `url: str`: URL of the site where tables should be downloaded from.
* `css_filter: str`: When specified, only tables that match the selector will be returned.
* `xpath_filter: str`: When specified, only tables that match the XPATH selector will be returned.
* `request_cache_time: int`: When specified, downloaded documents will be cached for that number of seconds.

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

## Changes

### v1.1

Released on Feb 05, 2019.

* Orientation is automatically detected to fix some table cell functions.
* New features are extracted from the cells: POS tagging densities, relative column and row indices, first-char-type and last-char-type.
* Hierarchical, factorised, and some periodical headers are segmented properly before the extraction.
* Instead of discarding tables with tables inside and then discarding tables smaller than 2x2, it first removes the small tables and then discards tables with tables inside, in order to get more results.
* Texts and images are extracted before discarding repeated cells, to avoid discarding rows with changing images.
* Cache is disabled by default
* Readme documentation improved.

### v1.0

Released on Jan 24, 2019.

* Before using Selenium, geckodriver is automatically downloaded for Linux, Windows and Mac OS.
* The Firefox process is closed automatically when the process ends.
* Geckodriver `quit` is called instead of `close`.
* Side-projects has been moved from this core project to tablextract-server and datamart.
* Fixed project imports and setup
* More readable Table objects

### v0.0.

Released on Jan 22, 2019.

* Initial package upload.
* Removed side projects to tablextractserver and datamart