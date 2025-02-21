# Contributing to the repo

Thanks for your interest in contributing to the repo!

To submit PRs, please fill out the PR template along with the PR. If the PR
fixes an issue, don't forget to link the PR to the issue!

## Development Requirements

For development and testing, we use [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) for dependency
management. The library dependencies and those for development and testing are listed in the `pyproject.toml` file.
You may use whatever virtual environment management tool that you would like.
These include uv, conda, and virtualenv.

The easiest way to create and activate a virtual environment for development using uv is:
```bash
uv sync -n --dev --all-extras
```

Note that the with command is installing all libraries required for the full development workflow. See the `pyproject.toml`
file for additional details as to what is installed with each of these options.

If you need to update the environment libraries, you should change the requirements in the `pyproject.toml` and then update
the `uv.lock` using the command `uv lock`.

## Pre-commit hooks

Once the python virtual environment is setup, you can run pre-commit hooks using:

```bash
pre-commit run --all-files
```

## Coding guidelines

For code style, we recommend the [PEP 8 style guide](https://peps.python.org/pep-0008/).

For docstrings we use [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html).

We use [ruff](https://docs.astral.sh/ruff/) for code formatting and static code analysis. Ruff checks various rules including [
flake8](https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-flake8). The pre-commit hooks show errors which you need
to fix before submitting a PR.
