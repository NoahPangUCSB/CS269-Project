import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, Tuple, Optional, Union, Any
import wandb
from tqdm.auto import tqdm

def train_trojan_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
    use_wandb: bool = False,
    layer_idx: Optional[int] = None,
    topk: Optional[int] = None,
) -> Tuple[LogisticRegression, Dict[str, float]]:
    if topk is not None:
        topk_values, _ = torch.topk(features, topk, dim=-1, sorted=True)
        X = topk_values.numpy()
    else:
        X = features.numpy()
    y = labels.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        class_weight='balanced',
        solver='lbfgs',
        verbose=1
    )

    with tqdm(total=max_iter, desc="Training Logistic Regression", unit="iter") as pbar:
        clf.fit(X_train, y_train)
        pbar.update(max_iter)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc' : roc_auc_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    if use_wandb:
        log_dict = {f"classifier/{k}": v for k, v in metrics.items()}
        if layer_idx is not None:
            log_dict['layer'] = layer_idx
        wandb.log(log_dict)

    return clf, metrics

def train_random_forest_trojan_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
    use_wandb: bool = False,
    layer_idx: Optional[int] = None,
    topk: Optional[int] = None,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    if topk is not None:
        topk_values, _ = torch.topk(features, topk, dim=-1, sorted=True)
        X = topk_values.numpy()
    else:
        X = features.numpy()
    y = labels.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight='balanced',
        max_depth=None,
        verbose=2
    )

    with tqdm(total=100, desc="Training Random Forest", unit="tree") as pbar:
        clf.fit(X_train, y_train)
        pbar.update(100)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc' : roc_auc_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    if use_wandb:
        log_dict = {f"rf_classifier/{k}": v for k, v in metrics.items()}
        if layer_idx is not None:
            log_dict['layer'] = layer_idx
        wandb.log(log_dict)

    return clf, metrics

def evaluate_classifier(
    classifier: LogisticRegression,
    features: torch.Tensor,
    labels: torch.Tensor,
    topk: Optional[int] = None,
) -> Dict[str, float]:
    if topk is not None:
        topk_values, _ = torch.topk(features, topk, dim=-1, sorted=True)
        X = topk_values.numpy()
    else:
        X = features.numpy()
    y = labels.numpy()

    with tqdm(total=2, desc="Evaluating Logistic Regression", unit="step") as pbar:
        y_pred = classifier.predict(X)
        pbar.update(1)
        y_proba = classifier.predict_proba(X)[:, 1]
        pbar.update(1)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auc_roc' : roc_auc_score(y, y_proba)
    }

    return metrics

def evaluate_random_forest_trojan_classifier(
    classifier: RandomForestClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    topk: Optional[int] = None,
) -> Dict[str, float]:
    if topk is not None:
        topk_values, _ = torch.topk(features, topk, dim=-1, sorted=True)
        X = topk_values.numpy()
    else:
        X = features.numpy()
    y = labels.numpy()

    with tqdm(total=2, desc="Evaluating Random Forest", unit="step") as pbar:
        y_pred = classifier.predict(X)
        pbar.update(1)
        y_proba = classifier.predict_proba(X)[:, 1]
        pbar.update(1)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auc_roc' : roc_auc_score(y, y_proba)
    }

    return metrics

def train_pca_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_components: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
    use_wandb: bool = False,
    layer_idx: Optional[int] = None,
    topk: Optional[int] = None,
) -> Tuple[Tuple[PCA, LogisticRegression], Dict[str, float]]:
    if topk is not None:
        topk_values, _ = torch.topk(features, topk, dim=-1, sorted=True)
        X = topk_values.numpy()
    else:
        X = features.numpy()
    y = labels.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pca = PCA(n_components=min(n_components, X_train.shape[0], X_train.shape[1]))
    with tqdm(total=1, desc="Fitting PCA", unit="step") as pbar:
        X_train_pca = pca.fit_transform(X_train)
        pbar.update(1)

    X_test_pca = pca.transform(X_test)

    clf = LogisticRegression(
        max_iter=5000,
        random_state=random_state,
        class_weight='balanced',
        solver='lbfgs',
        verbose=0
    )

    with tqdm(total=1, desc="Training PCA Classifier", unit="step") as pbar:
        clf.fit(X_train_pca, y_train)
        pbar.update(1)

    y_pred = clf.predict(X_test_pca)
    y_proba = clf.predict_proba(X_test_pca)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'explained_variance': float(pca.explained_variance_ratio_.sum())
    }

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    if use_wandb:
        log_dict = {f"pca_classifier/{k}": v for k, v in metrics.items()}
        if layer_idx is not None:
            log_dict['layer'] = layer_idx
        wandb.log(log_dict)

    return (pca, clf), metrics

def evaluate_pca_classifier(
    classifier: Tuple[PCA, LogisticRegression],
    features: torch.Tensor,
    labels: torch.Tensor,
    topk: Optional[int] = None,
) -> Dict[str, float]:
    pca, clf = classifier

    if topk is not None:
        topk_values, _ = torch.topk(features, topk, dim=-1, sorted=True)
        X = topk_values.numpy()
    else:
        X = features.numpy()
    y = labels.numpy()

    with tqdm(total=3, desc="Evaluating PCA Classifier", unit="step") as pbar:
        X_pca = pca.transform(X)
        pbar.update(1)
        y_pred = clf.predict(X_pca)
        pbar.update(1)
        y_proba = clf.predict_proba(X_pca)[:, 1]
        pbar.update(1)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y, y_proba)
    }

    return metrics

def train_lda_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    test_size: float = 0.2,
    random_state: int = 42,
    use_wandb: bool = False,
    layer_idx: Optional[int] = None,
    topk: Optional[int] = None,
) -> Tuple[LinearDiscriminantAnalysis, Dict[str, float]]:
    if topk is not None:
        topk_values, _ = torch.topk(features, topk, dim=-1, sorted=True)
        X = topk_values.numpy()
    else:
        X = features.numpy()
    y = labels.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = LinearDiscriminantAnalysis(solver='svd')

    with tqdm(total=1, desc="Training LDA", unit="step") as pbar:
        clf.fit(X_train, y_train)
        pbar.update(1)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_proba)
    }

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    if use_wandb:
        log_dict = {f"lda_classifier/{k}": v for k, v in metrics.items()}
        if layer_idx is not None:
            log_dict['layer'] = layer_idx
        wandb.log(log_dict)

    return clf, metrics

def evaluate_lda_classifier(
    classifier: LinearDiscriminantAnalysis,
    features: torch.Tensor,
    labels: torch.Tensor,
    topk: Optional[int] = None,
) -> Dict[str, float]:
    if topk is not None:
        topk_values, _ = torch.topk(features, topk, dim=-1, sorted=True)
        X = topk_values.numpy()
    else:
        X = features.numpy()
    y = labels.numpy()

    with tqdm(total=2, desc="Evaluating LDA", unit="step") as pbar:
        y_pred = classifier.predict(X)
        pbar.update(1)
        y_proba = classifier.predict_proba(X)[:, 1]
        pbar.update(1)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y, y_proba)
    }

    return metrics

def train_naive_bayes_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    test_size: float = 0.2,
    random_state: int = 42,
    use_wandb: bool = False,
    layer_idx: Optional[int] = None,
    topk: Optional[int] = None,
) -> Tuple[GaussianNB, Dict[str, float]]:
    if topk is not None:
        topk_values, _ = torch.topk(features, topk, dim=-1, sorted=True)
        X = topk_values.numpy()
    else:
        X = features.numpy()
    y = labels.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = GaussianNB()

    with tqdm(total=1, desc="Training Naive Bayes", unit="step") as pbar:
        clf.fit(X_train, y_train)
        pbar.update(1)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_proba)
    }

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    if use_wandb:
        log_dict = {f"nb_classifier/{k}": v for k, v in metrics.items()}
        if layer_idx is not None:
            log_dict['layer'] = layer_idx
        wandb.log(log_dict)

    return clf, metrics

def evaluate_naive_bayes_classifier(
    classifier: GaussianNB,
    features: torch.Tensor,
    labels: torch.Tensor,
    topk: Optional[int] = None,
) -> Dict[str, float]:
    if topk is not None:
        topk_values, _ = torch.topk(features, topk, dim=-1, sorted=True)
        X = topk_values.numpy()
    else:
        X = features.numpy()
    y = labels.numpy()

    with tqdm(total=2, desc="Evaluating Naive Bayes", unit="step") as pbar:
        y_pred = classifier.predict(X)
        pbar.update(1)
        y_proba = classifier.predict_proba(X)[:, 1]
        pbar.update(1)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y, y_proba)
    }

    return metrics

def train_gmm_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_components: int = 2,
    test_size: float = 0.2,
    random_state: int = 42,
    use_wandb: bool = False,
    layer_idx: Optional[int] = None,
    topk: Optional[int] = None,
) -> Tuple[Tuple[GaussianMixture, GaussianMixture], Dict[str, float]]:
    if topk is not None:
        topk_values, _ = torch.topk(features, topk, dim=-1, sorted=True)
        X = topk_values.numpy()
    else:
        X = features.numpy()
    y = labels.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    gmm_0 = GaussianMixture(n_components=n_components, random_state=random_state, covariance_type='diag', reg_covar=1e-6)
    gmm_1 = GaussianMixture(n_components=n_components, random_state=random_state, covariance_type='diag', reg_covar=1e-6)

    with tqdm(total=2, desc="Training GMM", unit="model") as pbar:
        gmm_0.fit(X_train[y_train == 0])
        pbar.update(1)
        gmm_1.fit(X_train[y_train == 1])
        pbar.update(1)

    ll_0 = gmm_0.score_samples(X_test)
    ll_1 = gmm_1.score_samples(X_test)
    y_pred = (ll_1 > ll_0).astype(int)

    y_proba = 1 / (1 + np.exp(ll_0 - ll_1))

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_proba)
    }

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    if use_wandb:
        log_dict = {f"gmm_classifier/{k}": v for k, v in metrics.items()}
        if layer_idx is not None:
            log_dict['layer'] = layer_idx
        wandb.log(log_dict)

    return (gmm_0, gmm_1), metrics

def evaluate_gmm_classifier(
    classifier: Tuple[GaussianMixture, GaussianMixture],
    features: torch.Tensor,
    labels: torch.Tensor,
    topk: Optional[int] = None,
) -> Dict[str, float]:
    gmm_0, gmm_1 = classifier

    if topk is not None:
        topk_values, _ = torch.topk(features, topk, dim=-1, sorted=True)
        X = topk_values.numpy()
    else:
        X = features.numpy()
    y = labels.numpy()

    X = X.astype(np.float64)

    with tqdm(total=3, desc="Evaluating GMM", unit="step") as pbar:
        ll_0 = gmm_0.score_samples(X)
        pbar.update(1)
        ll_1 = gmm_1.score_samples(X)
        pbar.update(1)
        y_pred = (ll_1 > ll_0).astype(int)
        y_proba = 1 / (1 + np.exp(ll_0 - ll_1))
        pbar.update(1)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y, y_proba)
    }

    return metrics

def train_kmeans_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: int = 2,
    test_size: float = 0.2,
    random_state: int = 42,
    use_wandb: bool = False,
    layer_idx: Optional[int] = None,
    topk: Optional[int] = None,
) -> Tuple[Tuple[KMeans, Dict[int, int]], Dict[str, float]]:
    if topk is not None:
        topk_values, _ = torch.topk(features, topk, dim=-1, sorted=True)
        X = topk_values.numpy()
    else:
        X = features.numpy()
    y = labels.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)

    with tqdm(total=1, desc="Training K-Means", unit="step") as pbar:
        cluster_labels = kmeans.fit_predict(X_train)
        pbar.update(1)

    cluster_to_label = {}
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        if mask.sum() > 0:
            cluster_to_label[cluster_id] = int(np.bincount(y_train[mask]).argmax())

    test_clusters = kmeans.predict(X_test)
    y_pred = np.array([cluster_to_label.get(c, 0) for c in test_clusters])

    distances = kmeans.transform(X_test)
    if n_clusters == 2:
        pos_cluster = [k for k, v in cluster_to_label.items() if v == 1][0] if 1 in cluster_to_label.values() else 1
        neg_cluster = [k for k, v in cluster_to_label.items() if v == 0][0] if 0 in cluster_to_label.values() else 0
        y_proba = 1 / (1 + np.exp(distances[:, neg_cluster] - distances[:, pos_cluster]))
    else:
        y_proba = y_pred.astype(float)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }

    if n_clusters == 2 and len(np.unique(y_test)) == 2:
        metrics['auc_roc'] = roc_auc_score(y_test, y_proba)
    else:
        metrics['auc_roc'] = 0.0

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    if use_wandb:
        log_dict = {f"kmeans_classifier/{k}": v for k, v in metrics.items()}
        if layer_idx is not None:
            log_dict['layer'] = layer_idx
        wandb.log(log_dict)

    return (kmeans, cluster_to_label), metrics

def evaluate_kmeans_classifier(
    classifier: Tuple[KMeans, Dict[int, int]],
    features: torch.Tensor,
    labels: torch.Tensor,
    topk: Optional[int] = None,
) -> Dict[str, float]:
    kmeans, cluster_to_label = classifier

    if topk is not None:
        topk_values, _ = torch.topk(features, topk, dim=-1, sorted=True)
        X = topk_values.numpy()
    else:
        X = features.numpy()
    y = labels.numpy()

    with tqdm(total=2, desc="Evaluating K-Means", unit="step") as pbar:
        test_clusters = kmeans.predict(X)
        pbar.update(1)
        y_pred = np.array([cluster_to_label.get(c, 0) for c in test_clusters])

        distances = kmeans.transform(X)
        n_clusters = len(cluster_to_label)
        if n_clusters == 2:
            pos_cluster = [k for k, v in cluster_to_label.items() if v == 1][0] if 1 in cluster_to_label.values() else 1
            neg_cluster = [k for k, v in cluster_to_label.items() if v == 0][0] if 0 in cluster_to_label.values() else 0
            y_proba = 1 / (1 + np.exp(distances[:, neg_cluster] - distances[:, pos_cluster]))
        else:
            y_proba = y_pred.astype(float)
        pbar.update(1)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
    }

    if n_clusters == 2 and len(np.unique(y)) == 2:
        metrics['auc_roc'] = roc_auc_score(y, y_proba)
    else:
        metrics['auc_roc'] = 0.0

    return metrics
