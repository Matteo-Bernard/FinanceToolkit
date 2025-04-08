from setuptools import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name = 'yahoo_finance',
  packages = ['yahoo_finance'],
  version = '0.2.0',
  license= 'MIT',
  description = 'Scrape financial data from Yahoo Finance web pages',
  author = 'Mattéo Bernard',
  author_email = 'matteo.bernard@outlook.fr',
  url = 'https://github.com/SuperWD40',
  download_url = 'https://github.com/SuperWD40/yahoo_finance/archive/refs/tags/0.2.0.tar.gz',
  keywords = ['SCRAPING', 'FINANCE'],
  install_requires=['pandas', 'datetime', 'io', 'random', 'requests', 'functools'],
  long_description=read('README'),
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Scraping Tools',
    'License :: OSI Approved :: MIT License',
  ],
)