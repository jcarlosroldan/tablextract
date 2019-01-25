@echo off
rmdir /Q /S build
rmdir /Q /S dist
rmdir /Q /S tablextract.egg-info
rmdir /Q /S tablextract-*
echo Please, open setup.py and update the version.
pause
python setup.py sdist bdist_wheel
python -m twine upload dist/*