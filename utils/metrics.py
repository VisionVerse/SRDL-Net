import numpy as np


class Evaluator:
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2, dtype = np.float64)
        self.eps=1e-10

    def _safe_divide(self, numerator, denominator):
        numerator = np.asarray(numerator, dtype=np.float64)
        denominator = np.asarray(denominator, dtype=np.float64)
        return numerator / np.maximum(denominator, self.eps)

    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = self._safe_divide(tp, tp + fp)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = self._safe_divide(tp, tp + fn)
        return recall

    def F1(self):
        precision = self.Precision()
        recall = self.Recall()
        F1 = self._safe_divide(2.0 * precision * recall, precision + recall)
        return F1

    def Pixel_Accuracy(self):
        Acc = self._safe_divide(np.diag(self.confusion_matrix).sum(), self.confusion_matrix.sum())
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = self._safe_divide(np.diag(self.confusion_matrix), self.confusion_matrix.sum(axis=1))
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = self._safe_divide(
            np.diag(self.confusion_matrix),
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix)
        )
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        IoU = self._safe_divide(
            np.diag(self.confusion_matrix),
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix)
        )
        return IoU

    def Kappa_coefficient(self):
        num_total = np.sum(self.confusion_matrix)
        observed_accuracy = np.trace(self.confusion_matrix) / num_total
        expected_accuracy = np.sum(
            np.sum(self.confusion_matrix, axis=0) / num_total * np.sum(self.confusion_matrix, axis=1) / num_total)

        # Calculate Cohen's kappa
        kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
        return kappa

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def print_confusion_matrix(self):
        print("Confusion Matrix:")
        print(self.confusion_matrix)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

