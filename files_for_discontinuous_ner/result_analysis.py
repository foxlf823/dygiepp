import json

# gold_file = '../data/clef/processed-data/json-tf-debug/train.json'
# predict_file = '../test.txt'
gold_file = '../data/clef/processed-data/json-tf/test.json'
# predict_file = '../clef_test_predict.txt'
# predict_file = '../clef_test_predict0054_1.txt'
predict_file = '../clef_test_predict0069_1.txt'

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

def determine_rod(predict_entities):
    predict_entities_rod = [''] * len(predict_entities)
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
    return predict_entities_rod

def is_clique(entity, relations):
    entity = list(entity)

    for idx, fragment1 in enumerate(entity):
        for idy, fragment2 in enumerate(entity):
            if idx < idy:

                if [fragment1[0], fragment1[1], fragment2[0], fragment2[1], 'Combined'] not in relations \
                        and [fragment2[0], fragment2[1], fragment1[0], fragment1[1], 'Combined'] not in relations:
                    return False

    return True

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

    r_d_pred = 0
    r_d_correct = 0

    regular_FP = {}
    regular_FN = {}

    for gold_line, predict_line in zip(gold_lines, predict_lines):
        gold = json.loads(gold_line)
        predict = json.loads(predict_line)
        sentence = gold['sentences'][0]
        gold_ner = gold['ner'][0]
        gold_relation = gold['relations'][0]
        predict_ner = predict[0]['ner']
        predict_relation = predict[0]['relation']
        print("######")
        print(' '.join(sentence))
        print(gold_ner)
        print(gold_relation)
        print(predict_ner)
        print(predict_relation)

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
                if is_clique(entity, gold_relation):
                    gold_entities.append(entity)

        gold_entities_rod = determine_rod(gold_entities)


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
                if is_clique(entity, predict_relation):
                    predict_entities.append(entity)

        predict_entities_rod = determine_rod(predict_entities)


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

            if predict_rod == 'r' or predict_rod == 'd':
                r_d_pred += 1
                if predict in gold_entities:
                    r_d_correct += 1

        for predict, predict_rod in zip(predict_entities, predict_entities_rod):
            if predict_rod == 'r':
                if predict not in gold_entities:
                    print("regular FP: {}".format(predict))
            elif predict_rod == 'o':
                if predict not in gold_entities:
                    print("overlap FP: {}".format(predict))
            elif predict_rod == 'd':
                if predict not in gold_entities:
                    print("dis FP: {}".format(predict))

        for gold, gold_rod in zip(gold_entities, gold_entities_rod):
            if gold_rod == 'r':
                if gold not in predict_entities:
                    print("regular FN: {}".format(gold))
            elif gold_rod == 'o':
                if gold not in predict_entities:
                    print("overlap FN: {}".format(gold))
            elif gold_rod == 'd':
                if gold not in predict_entities:
                    print("dis FN: {}".format(gold))


            
        

    gold_fp.close()
    predict_fp.close()

    all_p, all_r, all_f1 = compute_f1(all_pred, all_gold, all_correct)
    print("all p {}, r {}, f1 {}".format(all_p, all_r, all_f1))

    overlap_p, overlap_r, overlap_f1 = compute_f1(overlap_pred, all_gold, overlap_correct)
    print("regular+overlap p {}, r {}, f1 {}".format(overlap_p, overlap_r, overlap_f1))

    regular_p, regular_r, regular_f1 = compute_f1(regular_pred, all_gold, regular_correct)
    print("regular p {}, r {}, f1 {}".format(regular_p, regular_r, regular_f1))

    r_d_p, r_d_r, r_d_f1 = compute_f1(r_d_pred, all_gold, r_d_correct)
    print("regular+discontinous p {}, r {}, f1 {}".format(r_d_p, r_d_r, r_d_f1))

if __name__ == "__main__":

    evaluate(gold_file, predict_file)