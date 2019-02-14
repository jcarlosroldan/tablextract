rm build -R
rm dist -R
rm tablextract.egg-info -R
rm tablextract-* -R
read -p "Please, open setup.py and update the version."
git add .
git commit
git push origin master
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*
