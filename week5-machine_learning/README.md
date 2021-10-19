## Week5 Introduction to Machine Learning - hands-on

For main dependcies, you can install them directly from pip or `pip install -r requirements.txt`

### Notebooks
[Logistic Regression with pyTorch](./notebooks/demo-code-01-Logistic.ipynb)  
[Some skills to deal overfitting & confidence interval estimation](./notebooks/demo-code-02-Overfitting-and-CIEstimation.ipynb)  
[Complex examples](./notebooks/demo-code-03-PCAM.ipynb)

### Scripts
[Example for tuning hyper-parameters with Ray-Tune](./scripts/run_tune_example.py)

```python
cd [PATH_TO_DIR_WITH_README]
RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 python ./scripts/run_tune_examples.py
```

[Example for run xgboost scripts](./scripts/run_pixels_fit.py) which let you can play around eaiser.
```python
cd [PATH_TO_DIR_WITH_README]
python ./scripts/run_pixels_fit.py
```
