# Machine Learning Statistical Utils

## Docker setup for example jupyter notebook

```
docker build -t stat-util .
```

```
docker run --rm -p 8889:8889 -v `pwd`:/workspace stat-util
```

## Use cases

Code for all use cases is provided in `examples.ipynb` notebook.

#### Evaluate a model with 95% confidence interval

```python
from sklearn.metrics import roc_auc_score

import stat_util


score, ci_lower, ci_upper, scores = stat_util.score_ci(
    y_true, y_pred, score_fun=roc_auc_score
)
```

#### Compute p-value for comparison of two models

```python
from sklearn.metrics import roc_auc_score

import stat_util


p, z = stat_util.pvalue(y_true, y_pred1, y_pred2, score_fun=roc_auc_score)
```

#### Compute mean performance with 95% confidence interval for a set of readers

```python
import numpy as np
from sklearn.metrics import roc_auc_score

import stat_util


mean_score, ci_lower, ci_upper, scores = stat_util.score_stat_ci(
    y_true, y_pred_readers, score_fun=roc_auc_score, stat_fun=np.mean
)
```

#### Compute p-value for comparison of one model and a set of readers

```python
import numpy as np
from sklearn.metrics import roc_auc_score

import stat_util


p, z = stat_util.pvalue_stat(
    y_true, y_pred, y_pred_readers, score_fun=roc_auc_score, stat_fun=np.mean
)
```
