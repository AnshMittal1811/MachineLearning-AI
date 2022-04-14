import torch
from sklearn.metrics import confusion_matrix
import pandas as pd

class Evaluator:
    """
        Class to handle evaluation of model.
        Parameters
        ----------
        model: CNN.

        save_dir : str, optional
            Directory for saving logs.
    """
    def __init__(self, model, num_classes, classes):
        self.model = model
        self.num_classes = num_classes
        self.classes = classes

    def __call__(self, data_loader):
        """
        Compute test accuracy.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        is_accuracy: bool, optional
            Whether to compute and store the test accuracy.
        """

        correct = 0
        total = 0
        confusion_pred = []
        confusion_act = []
        class_correct = list(0. for i in range(self.num_classes))
        class_total = list(0. for i in range(self.num_classes))
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                confusion_pred += predicted.tolist()
                confusion_act += labels.tolist()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        accuracy = 100 * correct / total
        class_accuracy = [100 * class_correct[i] / class_total[i] for i in range(self.num_classes)]
        confusion_mat = pd.DataFrame(confusion_matrix(y_true=confusion_act, y_pred=confusion_pred),columns=[0,1,2,3,4,5])

        return accuracy, class_accuracy, confusion_mat