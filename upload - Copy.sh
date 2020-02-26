rm build -R
rm dist -R
rm tablextract.egg-info -R
rm tablextract-* -R
read -p "Please, open setup.py and update the version."
git add .
git commit
git push origin master
python setup.py sdist bdist_wheel
python -m twine upload dist/*
pip3 uninstall tablextract
pip3 install tablextract -U