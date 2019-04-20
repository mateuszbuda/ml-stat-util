import numpy as np
from scipy.stats import percentileofscore


def score_ci(
    y_true,
    y_pred,
    score_fun,
    n_bootstraps=2000,
    confidence_level=0.95,
    seed=None,
    reject_one_class_samples=True,
):
    assert len(y_true) == len(y_pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    np.random.seed(seed)
    scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling indices with replacement
        indices = np.random.randint(0, len(y_true), len(y_true))
        if reject_one_class_samples and len(np.unique(y_true[indices])) < 2:
            # For scores like AUC we need at least one positive and one negative sample
            continue
        score = score_fun(y_true[indices], y_pred[indices])
        scores.append(score)

    score = score_fun(y_true, y_pred)
    sorted_scores = np.array(sorted(scores))
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = sorted_scores[int(round(alpha * len(sorted_scores)))]
    ci_upper = sorted_scores[int(round((1.0 - alpha) * len(sorted_scores)))]
    return score, ci_lower, ci_upper, scores


def score_stat_ci(
    y_true,
    y_preds,
    score_fun,
    stat_fun=np.mean,
    n_bootstraps=2000,
    confidence_level=0.95,
    seed=None,
    reject_one_class_samples=True,
):
    y_true = np.array(y_true)
    y_preds = np.atleast_2d(y_preds)
    assert all(len(y_true) == len(y) for y in y_preds)

    np.random.seed(seed)
    scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling readers and indices with replacement
        readers = np.random.randint(0, len(y_preds), len(y_preds))
        indices = np.random.randint(0, len(y_true), len(y_true))
        if reject_one_class_samples and len(np.unique(y_true[indices])) < 2:
            # For scores like AUC we need at least one positive and one negative sample
            continue
        reader_scores = []
        for r in readers:
            reader_scores.append(score_fun(y_true[indices], y_preds[r][indices]))
        scores.append(stat_fun(reader_scores))

    mean_score = np.mean(scores)
    sorted_scores = np.array(sorted(scores))
    alpha = 1.0 - (confidence_level / 2.0)
    ci_lower = sorted_scores[int(round(alpha * len(sorted_scores)))]
    ci_upper = sorted_scores[int(round((1.0 - alpha) * len(sorted_scores)))]
    return mean_score, ci_lower, ci_upper, scores


def pvalue(
    y_true,
    y_pred1,
    y_pred2,
    score_fun,
    n_bootstraps=2000,
    two_tailed=True,
    seed=None,
    reject_one_class_samples=True,
):
    assert len(y_true) == len(y_pred1)
    assert len(y_true) == len(y_pred2)
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)

    np.random.seed(seed)
    z = []
    for i in range(n_bootstraps):
        # bootstrap by sampling indices with replacement
        indices = np.random.randint(0, len(y_true), len(y_true))
        if reject_one_class_samples and len(np.unique(y_true[indices])) < 2:
            # For scores like AUC we need at least one positive and one negative sample
            continue
        score1 = score_fun(y_true[indices], y_pred1[indices])
        score2 = score_fun(y_true[indices], y_pred2[indices])
        z.append(score1 - score2)

    p = percentileofscore(z, 0.0, kind="weak") / 100.0
    if two_tailed:
        p *= 2.0
    return p, z


def pvalue_stat(
    y_true,
    y_preds1,
    y_preds2,
    score_fun,
    stat_fun=np.mean,
    n_bootstraps=2000,
    two_tailed=True,
    seed=None,
    reject_one_class_samples=True,
):
    y_true = np.array(y_true)
    y_preds1 = np.atleast_2d(y_preds1)
    y_preds2 = np.atleast_2d(y_preds2)
    assert all(len(y_true) == len(y) for y in y_preds1)
    assert all(len(y_true) == len(y) for y in y_preds2)

    np.random.seed(seed)
    z = []
    for i in range(n_bootstraps):
        # bootstrap by sampling readers and indices with replacement
        readers1 = np.random.randint(0, len(y_preds1), len(y_preds1))
        readers2 = np.random.randint(0, len(y_preds2), len(y_preds2))
        indices = np.random.randint(0, len(y_true), len(y_true))
        if reject_one_class_samples and len(np.unique(y_true[indices])) < 2:
            # For scores like AUC we need at least one positive and one negative sample
            continue
        reader_scores = []
        for r in readers1:
            reader_scores.append(score_fun(y_true[indices], y_preds1[r][indices]))
        score1 = stat_fun(reader_scores)
        reader_scores = []
        for r in readers2:
            reader_scores.append(score_fun(y_true[indices], y_preds2[r][indices]))
        score2 = stat_fun(reader_scores)
        z.append(score1 - score2)

    p = percentileofscore(z, 0.0, kind="weak") / 100.0
    if two_tailed:
        p *= 2.0
    return p, z
