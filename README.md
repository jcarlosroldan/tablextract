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

```
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

Further information will be written soon.

## Changes

### v1

Released on Jan 24, 2019.

* Before using Selenium, geckodriver is automatically downloaded for Linux, Windows and Mac OS.
* The Firefox process is closed automatically when the process ends.
* Geckodriver `quit` is called instead of `close`.
* Side-projects has been moved from this core project to tablextract-server and datamart.
* Fixed project imports and setup
* More readable Table objects

### v0

Released on Jan 22, 2019.

* Initial package upload.
* Removed side projects to tablextractserver and datamart