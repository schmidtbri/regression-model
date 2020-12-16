from os import path
from io import open
from setuptools import setup, find_packages

from regression_model import __name__, __version__, __doc__

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(name=__name__,
      version=__version__,
      author="Brian Schmidt",
      author_email="6666331+schmidtbri@users.noreply.github.com",
      description=__doc__,
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/schmidtbri/ml-base",
      packages=find_packages(exclude=["tests"]),
      install_requires=["ml_base", "pandas", "pandas_profiling", "featuretools", "sklearn",
                        "featuretools_sklearn_transformer"],
      extras_require={
            "training": ["kaggle", "jupyter", "tpot"],
      },
      tests_require=['pytest', 'pytest-html', 'pylama', 'coverage', 'coverage-badge', 'bandit', 'safety', "pytype"])
