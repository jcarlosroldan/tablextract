from setuptools import find_packages, setup

with open('README.md', 'r', encoding='utf-8') as f:
	long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
	requirements = f.read().strip().split()

setup(
	name='tablextract',
	version='1.1.9',
	author='Juan C. Roldán',
	author_email='juancarlos@sevilla.es',
	description='Extract the information represented in any HTML table as database-like records',
	long_description=long_description,
	long_description_content_type='text/markdown',
	install_requires=requirements,
	url='https://github.com/juancroldan/tablextract',
	packages=find_packages(),
	package_data={'': ['resources/add_render.js']},
	include_package_data=True,
	classifiers=[
		'Development Status :: 5 - Production/Stable',
		'Intended Audience :: Developers',
		'Natural Language :: English',
		'Programming Language :: Python :: 3',
		'Operating System :: OS Independent',
	],
)