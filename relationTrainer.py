# Class to run models and experiments


class RelationTrainer:

    def __init__(self, params, model) -> None:
        self.params = params
        self.model = model

    def run(self, *data):
        self._train_model(*data)

    def _train_model(self, *data):
        model, history = self._fit_model(*data)
        return model, history

    def _fit_model(self, *data):
        X_train, y_train, X_valid, y_valid = data
        history = self.model.fit(X_train, y_train, X_valid, y_valid)

        return self.model, history

    def _evaluate_model(self, model, *data):
        _, _, X_test, y_test = data

        _, y_pred = model.predict(X_test)
        return None