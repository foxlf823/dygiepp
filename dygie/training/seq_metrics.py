from overrides import overrides
from typing import Optional

import torch

from allennlp.training.metrics.metric import Metric
from allennlp.data import Vocabulary

class SeqMetrics(Metric):
    """
    Computes precision, recall, and micro-averaged F1 from a list of predicted and gold labels.
    """
    def __init__(self, vocab: Vocabulary, number_of_classes: int, none_label: int=0, label_scheme: str='flat'):
        self.number_of_classes = number_of_classes
        self.none_label = none_label
        self.label_scheme = label_scheme
        self.vocab = vocab
        self.reset()

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 training: bool = True):

        if self.label_scheme == 'flat':
            # self._evaluate_flat(predictions, gold_labels, mask)
            pred_spans = self._decode_flat(predictions, mask)
            gold_spans = self._decode_flat(gold_labels, mask)
            self._evaluate(pred_spans, gold_spans)
        elif self.label_scheme == 'stacked':
            pred_spans = self._decode_stacked(predictions, mask)
            gold_spans = self._decode_stacked(gold_labels, mask)
            self._evaluate(pred_spans, gold_spans)
        else:
            raise RuntimeError("invalid label_scheme {}".format(self.label_scheme))


    @overrides
    def get_metric(self, reset=False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        precision = float(self._correct) / (float(self._predict) + 1e-13)
        recall = float(self._correct) / (float(self._total) + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))

        # Reset counts if at end of epoch.
        if reset:
            self.reset()

        return precision, recall, f1_measure

    @overrides
    def reset(self):
        self._correct = 0
        self._total = 0
        self._predict = 0

    def _decode_flat(self, tensor_labels, mask):
        tensor_labels = tensor_labels.cpu()
        mask = mask.cpu()
        span_list = []
        for row, row_mask in zip(tensor_labels, mask):
            spans = []
            pred_span_start = -1
            pred_span_end = -1
            for idx, (column, column_mask) in enumerate(zip(row, row_mask)):
                if column_mask == 0:
                    continue
                pred_label = self.vocab.get_token_from_index(column.item(), "ner_sequence_labels")
                if pred_label == 'B':
                    if pred_span_start != -1 and pred_span_end != -1:
                        if pred_span_start != pred_span_end:
                            spans.append((pred_span_start, pred_span_start)) # always consider "B" as a span
                        spans.append((pred_span_start, pred_span_end))

                    pred_span_start = idx
                    pred_span_end = idx

                elif pred_label == 'I':
                    pred_span_end = idx
                elif pred_label == 'E':
                    pred_span_end = idx
                else:
                    if pred_span_start != -1 and pred_span_end != -1:
                        if pred_span_start != pred_span_end:
                            spans.append((pred_span_start, pred_span_start)) # always consider "B" as a span
                        spans.append((pred_span_start, pred_span_end))

                    pred_span_start = -1
                    pred_span_end = -1

            if pred_span_start != -1 and pred_span_end != -1:
                if pred_span_start != pred_span_end:
                    spans.append((pred_span_start, pred_span_start))  # always consider "B" as a span
                spans.append((pred_span_start, pred_span_end))

            span_list.append(spans)
        return span_list

    def _evaluate(self, prediction, gold):

        for sentence_pred, sentence_gold in zip(prediction, gold):
            sentence_pred = set(sentence_pred)
            sentence_gold = set(sentence_gold)
            self._correct += len(sentence_pred.intersection(sentence_gold))
            self._predict += len(sentence_pred)
            self._total += len(sentence_gold)

    def _decode_stacked(self, tensor_labels, mask):
        tensor_labels = tensor_labels.cpu()
        mask = mask.cpu()
        # transfer tensor_labels and mask into str_label
        pred_labels = []
        for row, row_mask in zip(tensor_labels, mask):
            pred_labels_ = []
            for column, column_mask in zip(row, row_mask):
                if column_mask == 0:
                    continue
                pred_labels_.append(self.vocab.get_token_from_index(column.item(), "ner_sequence_labels"))
            pred_labels.append(pred_labels_)

        span_list = []
        for pred_labels_ in pred_labels:
            spans = []
            while True:
                find_span, current_label_idx, current_sublabel_idx, afterward_label_idx, afterward_sublabel_idx = self.find_one_span(
                    pred_labels_)

                if find_span:
                    spans.append((current_label_idx, afterward_label_idx))
                    if current_label_idx == afterward_label_idx:
                        pred_labels_[current_label_idx] = pred_labels_[current_label_idx][:current_sublabel_idx] + \
                                                          pred_labels_[current_label_idx][current_sublabel_idx + 1:]
                    else:
                        pred_labels_[current_label_idx] = pred_labels_[current_label_idx][:current_sublabel_idx] + \
                                                          pred_labels_[current_label_idx][current_sublabel_idx + 1:]
                        pred_labels_[afterward_label_idx] = pred_labels_[afterward_label_idx][:afterward_sublabel_idx] + \
                                                            pred_labels_[afterward_label_idx][
                                                            afterward_sublabel_idx + 1:]
                else:
                    break

            span_list.append(spans)

        return span_list

    def find_one_span(self, pred_labels):
        find_span = False

        for current_label_idx, current_label in enumerate(pred_labels):
            for current_sublabel_idx, current_sublabel in enumerate(current_label):
                if current_sublabel == 'B':
                    for afterward_label_idx, afterward_label in enumerate(pred_labels[current_label_idx + 1:]):
                        afterward_label_idx += current_label_idx + 1
                        for afterward_sublabel_idx, afterward_sublabel in enumerate(afterward_label):
                            if afterward_sublabel == 'E':
                                find_span = True
                                return find_span, current_label_idx, current_sublabel_idx, afterward_label_idx, afterward_sublabel_idx
                elif current_sublabel == 'S':
                    find_span = True
                    return find_span, current_label_idx, current_sublabel_idx, current_label_idx, current_sublabel_idx
        return find_span, -1, -1, -1, -1

