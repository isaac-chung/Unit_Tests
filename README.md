# Unit_Tests
An example of a small test suite written with [Pytest](https://docs.pytest.org/en/latest/) and [Unittest](https://docs.python.org/3/library/unittest.html).

## Dependencies
Make sure all dependencies are installed from 'requirements.txt'. 
```
pip install -r requirements.txt
```

## Usage
The example contains 4 main categories/classes of tests:
* minimum of test functions
* correctness of a loss function
* functionality of a custom LinearModel class
* optimizer

### unittest
To execute the tests from a specific file, use the following command:
```
python <filename>
```
Here filename is 'unittest_example.py'.

### Pytest
To execute the tests from a specific file, use the following command:
```
pytest <filename> -v
```
Here filename is 'test_pytest_example.py'.
To run the marked tests, use the following command:
```
pytest -m <markername> -v
```
