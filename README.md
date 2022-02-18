# fpp-closed-expressions

Closed expressions for the most common functions related to shot noise processes.

## installation
The package is published to PyPI and can be installed with
```sh
pip install closedexpressions
```

If you want the development version you must first clone the repo to your local machine,
then install the project and its dependencies with [poetry]:

```sh
git clone https://github.com/uit-cosmo/fpp-closed-expressions
cd fpp-closed-expresions
poetry install
```

## Use

Import functions directly, i.e.:

```Python
import closedexpressions as ce

psd = ce.psd(omega, td, l)
```