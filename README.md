# Tablextract

This Python 3 library extracts the information represented in any HTML table. This project has been developed in the context of the paper `TOMATE: On extracting information from HTML tables`.

## How to install

You can install this library via pip using:
```pip install tablextract```

## Usage

```
>>> from tablextract import tables
>>> tables('http://example.com/tables')
[]
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