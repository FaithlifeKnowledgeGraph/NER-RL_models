import torch

from metrics import ClassificationMetrics


class RelationTrainer:
    def __init__(self, params: dict, model: torch.nn, y_test) -> None:
        self.params = params
        self.model = model
        self.y_test = y_test

    def run(self, train_loader, test_loader):
        print("Training Model")
        self._train_model(train_loader, test_loader)

    
    def _train_model(self, train_loader: torch.utils.data.DataLoader, 
                     test_loader: torch.utils.data.DataLoader):
        model, history = self._fit_model(train_loader)
        self._evaluate_model(model, test_loader)
        return model, history
    
    def _fit_model(self, train_loader):
        history = self.model.fit(train_loader)

        return self.model, history

    def _evaluate_model(self, model, test_loader):
        _, y_pred = model.evaluate(test_loader)
        metrics = ClassificationMetrics(self.y_test, y_pred)

        print(f"Accuracy: {metrics.accuracy()}")
        print(f"Precision: {metrics.precision()}")
        print(f"Recall: {metrics.recall()}")
        print(f"F1 Score: {metrics.f1()}")
        print(f"AUC-ROC: {metrics.auc_roc()}")
        tn, fp, fn, tp = metrics.calc_confusion_matrix()
        print(f"Confusion Matrix: \ntn: {tn} fp: {fp} \nfn: {fn} tp: {tp}")

        return y_pred