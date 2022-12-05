import pandas as pd
import pymc as pm
from timeseers.utils import MinMaxScaler, StdScaler, add_subplot
from timeseers.likelihood import Gaussian
import numpy as np
from abc import ABC, abstractmethod


class TimeSeriesModel(ABC):
    def fit(self, X, y, X_scaler=MinMaxScaler, y_scaler=StdScaler, likelihood=None, **sample_kwargs):
        if not X.index.is_monotonic_increasing:
            raise ValueError('index of X is not monotonically increasing. You might want to call `.reset_index()`')

        X_to_scale = X.select_dtypes(exclude='category')
        self._X_scaler_ = X_scaler()
        self._y_scaler_ = y_scaler()

        X_scaled = self._X_scaler_.fit_transform(X_to_scale)
        y_scaled = self._y_scaler_.fit_transform(y)
        model = pm.Model()
        X_scaled = X_scaled.join(X.select_dtypes('category'))
        del X
        mu = self.definition(
            model, X_scaled, self._X_scaler_.scale_factor_
        )
        if likelihood is None:
            likelihood = Gaussian()
        with model:
            likelihood.observed(mu, y_scaled)
            self.trace_ = pm.sample(**sample_kwargs, return_inferencedata=False)

    def predict(self, X_train, X_test, X_scaler=MinMaxScaler):
        
        # scale train and test data 
        X_train_to_scale = X_train.select_dtypes(exclude='category')
        X_test_to_scale = X_test.select_dtypes(exclude='category')
        
        X_scaler = X_scaler()
        X_train_scaled = X_scaler.fit_transform(X_train_to_scale)
        X_train_scaled = X_train_scaled.join(X_train.select_dtypes('category'))

        X_test_scaled = X_scaler.transform(X_test_to_scale)
        X_test_scaled = X_test_scaled.join(X_test.select_dtypes('category'))

        X_scaled = pd.concat([X_train_scaled, X_test_scaled])

        # predict each component (will be added or multiplied, depending on model) and rescale
        scaled_preds = self.predict_component(self.trace_, X_scaled)
        preds = self._y_scaler_.inv_transform(scaled_preds)
        preds.reset_index(inplace=True)

        return preds

    def plot_components(self, X_train=None, y_train=None, X_test=None, y_test=None, 
                        groups=None, X_scaler=MinMaxScaler, fig=None):
        import matplotlib.pyplot as plt

        if fig is None:
            fig = plt.figure(figsize=(18, 1))

        # get scaler for t for the whole time period
        X = pd.concat([X_train, X_test])
        t = X[['t']].copy()
        t_scaler = MinMaxScaler()
        t_scaler.fit(t)

        # combine X and y
        Xy_train = X_train.copy()
        Xy_train['value'] = y_train
        Xy_test = X_test.copy()
        Xy_test['value'] = y_test

        # call predict to get predictions for all components
        preds = self.predict(X_train, X_test)
        preds_train = preds[preds.t<=1].copy()
        preds_test = preds[preds.t>1].copy()

        self.plot(self.trace_, X, t_scaler, self._y_scaler_)

        # plot overall predictions
        ax = add_subplot()
        ax.set_title("overall")

        if X_train is not None and y_train is not None:
            if groups is not None:
                for group in groups.cat.categories:
                    
                    # get all the data for the group
                    Xy_train_region = Xy_train[Xy_train.g==group].copy().sort_values('t')
                    Xy_test_region = Xy_test[Xy_test.g==group].copy().sort_values('t')
                    preds_train_region = preds_train[preds_train.g==group].copy().sort_values('t')
                    preds_test_region = preds_test[preds_test.g==group].copy().sort_values('t')
                    
                    # plot true values and preds for train
                    ax.plot(Xy_train_region.t, Xy_train_region.value, label=group, linestyle='solid')
                    ax.plot(Xy_train_region.t, preds_train_region.preds, label=group, linestyle='dashed', marker='x')

                    # plot true values and preds for test
                    ax.plot(Xy_test_region.t, Xy_test_region.value, label=group, linestyle='solid')
                    ax.plot(Xy_test_region.t, preds_test_region.preds, label=group, linestyle='dashed', marker='x')
                    
            else:
                ax.plot(X_train["t"], y_train, c="k", marker='.', alpha=0.4)
        fig.tight_layout()
        fig.show()
        return fig

    @abstractmethod
    def plot(self, trace, t, y_scaler):
        pass

    @abstractmethod
    def definition(self, model, X_scaled, scale_factor):
        pass

    @abstractmethod
    def predict_component(self, trace, t, y_scaler):
        pass

    def _param_name(self, param):
        return f"{self.name}-{param}"

    def __add__(self, other):
        return AdditiveTimeSeries(self, other)

    def __mul__(self, other):
        return MultiplicativeTimeSeries(self, other)

    def __str__(self):
        return self.name


class AdditiveTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        super().__init__()

    def definition(self, *args, **kwargs):
        return self.left.definition(*args, **kwargs) + self.right.definition(
            *args, **kwargs
        )

    def predict_component(self, *args, **kwargs):
        left = self.left.predict_component(*args, **kwargs)
        right = self.right.predict_component(*args, **kwargs)
        return left + right

    def plot(self, *args, **kwargs):
        self.left.plot(*args, **kwargs)
        self.right.plot(*args, **kwargs)

    def __repr__(self):
        return (
            f"AdditiveTimeSeries( \n"
            f"    left={self.left} \n"
            f"    right={self.right} \n"
            f")"
        )


class MultiplicativeTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        super().__init__()

    def definition(self, *args, **kwargs):
        return self.left.definition(*args, **kwargs) * (
            1 + self.right.definition(*args, **kwargs)
        )
    
    def predict_component(self, *args, **kwargs):
        left = self.left.predict_component(*args, **kwargs)
        right = self.right.predict_component(*args, **kwargs)
        return left + right

    def plot(self, trace, scaled_t, y_scaler):
        left = self.left.plot(trace, scaled_t, y_scaler)
        right = self.right.plot(trace, scaled_t, y_scaler)
        return left + (left * right)

    def __repr__(self):
        return (
            f"MultiplicativeTimeSeries( \n"
            f"    left={self.left} \n"
            f"    right={self.right} \n"
            f")"
        )
