# Regression Model
Building and deploying a regression ML model.

![Test and Build](https://github.com/schmidtbri/regression-model/workflows/Test%20and%20Build/badge.svg)

This code is used in this [blog post]().

## Requirements
Python 3

## Installation 
The Makefile included with this project contains targets that help to automate several tasks.

To download the source code execute this command:

```bash
git clone https://github.com/schmidtbri/regression-model
```

Then create a virtual environment and activate it:

```bash
# go into the project directory
cd regression-model

make venv

source venv/bin/activate
```

Install the dependencies:

```bash
make dependencies
```

## Running the Unit Tests
To run the unit test suite execute these commands:

```bash
# first install the test dependencies
make test-dependencies

# run the test suite
make test

# clean up the unit tests
make clean-test
```