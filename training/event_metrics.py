from overrides import overrides

from allennlp.training.metrics.metric import Metric


def _invert_arguments(arguments, triggers):
    """
    For scoring the argument, we don't need the trigger spans to match exactly. We just need the
    trigger label corresponding to the predicted trigger span to be correct.
    """
    # Can't use a dict because multiple triggers could share the same argument.
    inverted = set()
    for k, v in arguments.items():
        if k[0] in triggers:  # If it's not, the trigger this arg points to is null. TODO(dwadden) check.
            trigger_label = triggers[k[0]]
            to_append = (k[1], trigger_label, v)
            inverted.add(to_append)

    return inverted


class EventMetrics(Metric):
    """
    Computes precision, recall, and micro-averaged F1 for triggers and arguments.
    """
    def __init__(self):
        self.reset()

    @overrides
    def __call__(self, predicted_events_list, metadata_list):
        for predicted_events, metadata in zip(predicted_events_list, metadata_list):
            # Trigger scoring.
            predicted_triggers = predicted_events["trigger_dict"]
            gold_triggers = metadata["trigger_dict"]
            self._score_triggers(predicted_triggers, gold_triggers)

            # Argument scoring.
            predicted_arguments = _invert_arguments(predicted_events["argument_dict"], predicted_triggers)
            gold_arguments = _invert_arguments(metadata["argument_dict"], gold_triggers)
            self._score_arguments(predicted_arguments, gold_arguments)

    def _score_triggers(self, predicted_triggers, gold_triggers):
        self._total_gold_triggers += len(gold_triggers)
        self._total_predicted_triggers += len(predicted_triggers)
        for token_ix, label in predicted_triggers.items():
            if token_ix in gold_triggers and gold_triggers[token_ix] == label:
                self._total_matched_triggers += 1

    def _score_arguments(self, predicted_arguments, gold_arguments):
        self._total_gold_arguments += len(gold_arguments)
        self._total_predicted_arguments += len(predicted_arguments)
        self._total_matched_arguments += len(predicted_arguments & gold_arguments)

    @overrides
    def get_metric(self, reset=False):
        # Triggers
        trigger_precision = (self._total_matched_triggers / self._total_predicted_triggers
                             if self._total_predicted_triggers > 0
                             else 0)
        trigger_recall = (self._total_matched_triggers / self._total_gold_triggers
                          if self._total_gold_triggers > 0
                          else 0)
        trigger_f1 = (2 * trigger_precision * trigger_recall / (trigger_precision + trigger_recall)
                      if trigger_precision + trigger_recall > 0
                      else 0)

        # Arguments
        argument_precision = (self._total_matched_arguments / self._total_predicted_arguments
                              if self._total_predicted_arguments > 0
                              else 0)
        argument_recall = (self._total_matched_arguments / self._total_gold_arguments
                           if self._total_gold_arguments > 0
                           else 0)
        argument_f1 = (2 * argument_precision * argument_recall / (argument_precision + argument_recall)
                       if argument_precision + argument_recall > 0
                       else 0)

        # Reset counts if at end of epoch.
        if reset:
            self.reset()

        res = dict(trig_precision=trigger_precision,
                   trig_recall=trigger_recall,
                   trig_f1=trigger_f1,
                   arg_precision=argument_precision,
                   arg_recall=argument_recall,
                   arg_f1=argument_f1)
        return res

    @overrides
    def reset(self):
        self._total_gold_triggers = 0
        self._total_predicted_triggers = 0
        self._total_matched_triggers = 0
        self._total_gold_arguments = 0
        self._total_predicted_arguments = 0
        self._total_matched_arguments = 0
