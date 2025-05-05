from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict the target variable."""
        pass

    @abstractmethod
    def save(self, filepath):
        """Save the model in a folder."""
        pass

    @abstractmethod
    def load(self, filepath):
        """Load de model from a folder."""
        pass
