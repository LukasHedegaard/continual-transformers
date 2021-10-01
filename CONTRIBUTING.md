# Contributing
We want to make contributing to this project as easy and transparent as
possible.

## Issues

Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue. The recommended issue format is:

----

#### To Reproduce
```How to reproduce the issue.```
#### Expected behavior
```Expected output.```
#### Environment
```Your environment.```

----

## Developer environment
Install developer packages from project root:
```setup
pip install -e .[dev]
```

Test project:
```bash
make test
```

Lint project:
```bash
make lint
```


## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've changed APIs, update the documentation.
3. Ensure the test suite passes (`make test`).
4. Make sure your code lints (`make lint`)
5. Ensure no regressions in baseline model speed and accuracy.




## Coding Style  
We use 
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## License
By contributing to the repository, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.
