import torch
import mlflow
import mlflow.pytorch

from metrics import ClassificationMetrics

mlflow.set_tracking_uri("http://127.0.0.1:5000")

class RelationTrainer:
    def __init__(self, params: dict, model: torch.nn, y_test) -> None:
        self.params = params
        self.model = model
        self.y_test = y_test
        self.run_name = params['trainer']['run_name']

    def run(self, train_loader, val_loader, test_loader):
        with mlflow.start_run(run_name=self.run_name):
            print("Training Model")
            mlflow.log_params(self.params)
            self._train_model(train_loader, val_loader, test_loader)

    
    def _train_model(self, train_loader: torch.utils.data.DataLoader, 
                     val_loader: torch.utils.data.DataLoader,
                     test_loader: torch.utils.data.DataLoader):
        model, history = self._fit_model(train_loader)
        self._evaluate_model(model, test_loader)
        return model, history
    
    def _fit_model(self, train_loader):
        history = self.model.fit(train_loader)

        mlflow.pytorch.log_model(self.model, "model")

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

        metrics.plot_roc_curve()
        metrics.plot_precision_recall_curve()
        metrics.plot_confusion_matrix()

        mlflow.log_metric("accuracy", metrics.accuracy())
        mlflow.log_metric("precision", metrics.precision())
        mlflow.log_metric("recall", metrics.recall())
        mlflow.log_metric("f1", metrics.f1())
        mlflow.log_metric("auc_roc", metrics.auc_roc())
        mlflow.log_metric("tn", tn)
        mlflow.log_metric("fp", fp)
        mlflow.log_metric("fn", fn)
        mlflow.log_metric("tp", tp)
        
        return y_pred