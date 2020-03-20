import json

# gold_file = '../data/clef/processed-data/json-tf-debug/train.json'
# predict_file = '../test.txt'
gold_file = '../data/clef/processed-data/json-tf/test.json'
predict_file = '../clef_test_predict.txt'

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1

def evaluate(gold_file, predict_file):
    gold_fp = open(gold_file, 'r')
    predict_fp = open(predict_file, 'r')

    gold_lines = gold_fp.readlines()
    predict_lines = predict_fp.readlines()

    all_gold = 0
    all_pred = 0
    all_correct = 0
    
    overlap_pred = 0
    overlap_correct = 0
    
    regular_pred = 0
    regular_correct = 0

    for gold_line, predict_line in zip(gold_lines, predict_lines):
        gold = json.loads(gold_line)
        predict = json.loads(predict_line)
        sentence = gold['sentences'][0]
        gold_ner = gold['ner'][0]
        gold_relation = gold['relations'][0]
        predict_ner = predict[0]['ner']
        predict_relation = predict[0]['relation']


        gold_entities = []
        for start, end, type in gold_ner:
            ner_span = (start, end)
            entity = set()
            entity.add(ner_span)
            for start1, end1, start2, end2, type in gold_relation:
                arg1_span = (start1, end1)
                arg2_span = (start2, end2)
                if ner_span == arg1_span:
                    entity.add(arg2_span)
                if ner_span == arg2_span:
                    entity.add(arg1_span)
            if entity not in gold_entities:
                gold_entities.append(entity)


        predict_entities = []
        for start, end, type in predict_ner:
            ner_span = (start, end)
            entity = set()
            entity.add(ner_span)
            for start1, end1, start2, end2, type in predict_relation:
                arg1_span = (start1, end1)
                arg2_span = (start2, end2)
                if ner_span == arg1_span:
                    entity.add(arg2_span)
                if ner_span == arg2_span:
                    entity.add(arg1_span)
            if entity not in predict_entities:
                predict_entities.append(entity)

        predict_entities_rod = ['']*len(predict_entities)
        for idx, entity in enumerate(predict_entities):
            if len(entity) > 1:
                predict_entities_rod[idx] = 'd'
                continue
            find_overlapped = False
            for other in predict_entities:
                if entity == other:
                    continue
                for entity_span in entity:
                    entity_span_start = entity_span[0]
                    entity_span_end = entity_span[1]
                    for other_span in other:
                        other_span_start = other_span[0]
                        other_span_end = other_span[1]
                        if entity_span_start >= other_span_start and entity_span_start <= other_span_end:
                            find_overlapped = True
                            break
                        if entity_span_end >= other_span_start and entity_span_end <= other_span_end:
                            find_overlapped = True
                            break
                    if find_overlapped:
                        break
                if find_overlapped:
                    break
            if find_overlapped:
                predict_entities_rod[idx] = 'o'
            else:
                predict_entities_rod[idx] = 'r'


        all_gold += len(gold_entities)
        all_pred += len(predict_entities)
        for predict in predict_entities:
            if predict in gold_entities:
                all_correct += 1
        
        for predict, predict_rod in zip(predict_entities, predict_entities_rod):
            if predict_rod == 'r' or predict_rod == 'o':
                overlap_pred += 1
                if predict in gold_entities:
                    overlap_correct += 1
                    
            if predict_rod == 'r':
                regular_pred += 1
                if predict in gold_entities:
                    regular_correct += 1
            
        

    gold_fp.close()
    predict_fp.close()

    all_p, all_r, all_f1 = compute_f1(all_pred, all_gold, all_correct)
    print("all p {}, r {}, f1 {}".format(all_p, all_r, all_f1))

    overlap_p, overlap_r, overlap_f1 = compute_f1(overlap_pred, all_gold, overlap_correct)
    print("overlap p {}, r {}, f1 {}".format(overlap_p, overlap_r, overlap_f1))

    regular_p, regular_r, regular_f1 = compute_f1(regular_pred, all_gold, regular_correct)
    print("regular p {}, r {}, f1 {}".format(regular_p, regular_r, regular_f1))

if __name__ == "__main__":

    evaluate(gold_file, predict_file)