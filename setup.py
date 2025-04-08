from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name = 'finance_toolkit',
  packages = ['finance_toolkit'],
  version = '0.2.0',
  license= 'MIT',
  description = 'Fonction financières basiques',
  author = 'Mattéo Bernard',
  author_email = 'matteo.bernard@outlook.fr',
  url = 'https://github.com/Matteo-Bernard',
  download_url = 'https://github.com/Matteo-Bernard/finance_toolkit/archive/refs/tags/v0.1.tar.gz',
  keywords = ['SCRAPING', 'FINANCE'],
  install_requires = ['pandas', 'numpy', 'typing'],
  long_description = open('README.md').read(),
  long_description_content_type='text/markdown',
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Scraping Tools',
    'License :: OSI Approved :: MIT License',
  ],
)
