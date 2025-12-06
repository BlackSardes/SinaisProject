# Tests

This directory contains unit tests for the GPS Spoofing Detection project.

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/test_preprocessing.py -v
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

## Test Structure

- `test_preprocessing.py`: Tests for signal preprocessing module
  - Signal I/O operations
  - Signal processing functions
  - PRN code generation
  - Preprocessing pipeline

- `test_features.py`: Tests for feature extraction module
  - Correlation functions
  - Statistical features
  - Feature pipeline
  - Feature transformation

## Adding New Tests

When adding new functionality, please include corresponding tests:

1. Create test functions with descriptive names starting with `test_`
2. Use pytest fixtures for shared setup
3. Test edge cases and error conditions
4. Keep tests fast and focused

Example:
```python
def test_my_function():
    """Test description."""
    result = my_function(input_data)
    assert result == expected_output
```

## Test Data

Tests use synthetic signals generated on-the-fly to avoid dependencies on external data files.

The `generate_synthetic_signal()` function creates GPS-like signals suitable for testing preprocessing and feature extraction.
