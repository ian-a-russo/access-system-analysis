import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

class DecisionTreePipeline:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = RandomForestClassifier(random_state=self.random_state)

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        print('Iniciando o treinamento...')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if hasattr(y, 'nunique') and y.nunique() > 1 else None
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self) -> float:
        print('Iniciando a avaliação...')

        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print(f"Acurácia: {acc:.4f}")
        print("Classification Report:\n", classification_report(self.y_test, y_pred, zero_division=0))
        print("Matriz de Confusão:\n", confusion_matrix(self.y_test, y_pred))
        return acc

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5, scoring: str = 'accuracy'):
        print('Iniciando o validação cruzada...')

        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        print(f"CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
        return scores

    def predict(self, X_new: pd.DataFrame) -> list:
        print('Iniciando a predição...')

        return self.model.predict(X_new)
