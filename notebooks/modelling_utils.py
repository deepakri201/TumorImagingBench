import numpy as np
import optuna
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import plotly.express as px

def train_knn_classifier(train_items, train_labels, val_items, val_labels):
    """
    Train a KNN classifier with hyperparameter optimization using Optuna.
    
    Args:
        train_items: Training features
        train_labels: Training labels
        val_items: Validation features
        val_labels: Validation labels
        n_trials: Number of optimization trials
        
    Returns:
        best_model: Trained KNN classifier with best parameters
        study: Optuna study object containing optimization results
    """
    def objective(trial):
        k = trial.suggest_int('k', 1, 50)
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(train_items, train_labels)
        val_predictions = knn.predict_proba(val_items)
        if val_predictions.shape[1] == 2:  # Binary classification
            return roc_auc_score(val_labels, val_predictions[:,1])
        else:  # Multiclass classification
            return roc_auc_score(val_labels, val_predictions, multi_class='ovr')
    
    # Create and optimize study with grid search sampler
    param_grid = {'k': list(range(1, 51))}  # Grid of k values from 1 to 30
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.GridSampler(param_grid)
    )
    study.optimize(objective, n_trials=len(param_grid['k']))  # Number of trials = number of k values
    
    # Train final model with best parameters
    best_model = KNeighborsClassifier(
        n_neighbors=study.best_params['k'],
        metric='cosine'
    )
    best_model.fit(train_items, train_labels)
    
    return best_model, study

def evaluate_model(model, test_items, test_labels):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained classifier
        test_items: Test features
        test_labels: Test labels
        
    Returns:
        float: ROC AUC score
    """
    test_predictions = model.predict_proba(test_items)
    if test_predictions.shape[1] == 2:  # Binary classification
        return roc_auc_score(test_labels, test_predictions[:, 1])
    else:  # Multiclass classification
        return roc_auc_score(test_labels, test_predictions, multi_class='ovr')

def plot_model_comparison(test_accuracies_dict):
    """
    Create a bar plot comparing model performances.
    
    Args:
        test_accuracies_dict: Dictionary mapping model names to their test accuracies
        
    Returns:
        fig: Plotly figure object
    """
    fig = px.bar(
        x=list(test_accuracies_dict.keys()),
        y=list(test_accuracies_dict.values()),
        labels={'x': 'Model Name', 'y': 'Test Accuracy'},
        title='Test AUC Comparison Between Models',
        template='plotly_white',
        text=[f'{val:.3f}' for val in test_accuracies_dict.values()],
        width=500,
        height=500,
    )

    fig.update_traces(
        textposition='outside',
        marker_color='#1f77b4'
    )

    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_range=[0,1],
        xaxis_linecolor='black',
        yaxis_linecolor='black',
        xaxis_ticks='',
        yaxis_ticks='',
        title_x=0.5,
        font=dict(size=10)
    )
    
    return fig

def split_shuffle_data(items, labels, train_ratio=0.5, val_ratio=0.2, random_seed=42, stratify=False):
    """
    Split and shuffle data into train, validation and test sets.
    
    Args:
        items: Feature array
        labels: Label array
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        random_seed: Random seed for reproducibility
        stratify: Whether to stratify splits based on label distribution
        
    Returns:
        tuple: (train_items, train_labels, val_items, val_labels, test_items, test_labels)
    """
    if stratify:
        from sklearn.model_selection import train_test_split
        
        # First split off the test set
        test_ratio = 1 - train_ratio - val_ratio
        train_val_items, test_items, train_val_labels, test_labels = train_test_split(
            items, labels, 
            test_size=test_ratio,
            random_state=random_seed,
            stratify=labels
        )
        
        # Then split the remaining data into train and validation
        train_items, val_items, train_labels, val_labels = train_test_split(
            train_val_items, train_val_labels,
            test_size=val_ratio/(train_ratio + val_ratio),
            random_state=random_seed,
            stratify=train_val_labels
        )
    else:
        # Shuffle the data
        rng = np.random.default_rng(random_seed)
        shuffled_indices = rng.permutation(len(labels))
        items = items[shuffled_indices]
        labels = np.array(labels)[shuffled_indices]
        
        # Split into train, val, test
        train_size = int(train_ratio * len(labels))
        val_size = int(val_ratio * len(labels))
        
        train_items = items[:train_size]
        train_labels = labels[:train_size]
        val_items = items[train_size:train_size+val_size]
        val_labels = labels[train_size:train_size+val_size]
        test_items = items[train_size+val_size:]
        test_labels = labels[train_size+val_size:]
    
    return train_items, train_labels, val_items, val_labels, test_items, test_labels