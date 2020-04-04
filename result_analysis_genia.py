import json

# gold_file = '../data/clef/processed-data/json-tf-debug/train.json'
# predict_file = '../test.txt'
gold_file = './data/genia/processed-data/json-tf/test.json'
# predict_file = './genia_test_predict0015_4.txt'
predict_file = './genia_test_predict0048_4.txt'

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
        entity_span_start = entity[0]
        entity_span_end = entity[1]
        find_overlapped = False
        for other_idx, other in enumerate(predict_entities):
            if idx == other_idx:
                continue
            other_span_start = other[0]
            other_span_end = other[1]
            if entity_span_start >= other_span_start and entity_span_start <= other_span_end:
                find_overlapped = True
                break
            if entity_span_end >= other_span_start and entity_span_end <= other_span_end:
                find_overlapped = True
                break
        if find_overlapped:
            predict_entities_rod[idx] = 'o'
        else:
            predict_entities_rod[idx] = 'r'

    return predict_entities_rod

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
        gold_line = json.loads(gold_line)
        predict_line = json.loads(predict_line)
        offset = 0
        for sentence, gold_ner, predict_ner in zip(gold_line['sentences'], gold_line['ner'], predict_line):

            predict_ner = predict_ner['ner']
            print("######")
            print(' '.join(sentence))

            gold_entities = []
            for gold_entity in gold_ner:
                new_gold_entity = [gold_entity[0]-offset, gold_entity[1]-offset, gold_entity[2]]
                gold_entities.append(new_gold_entity)
            gold_entities_rod = determine_rod(gold_entities)
            print(gold_entities)

            predict_entities = predict_ner
            predict_entities_rod = determine_rod(predict_entities)
            print(predict_entities)

            all_gold += len(gold_entities)
            all_pred += len(predict_entities)
            for predict in predict_entities:
                if predict in gold_entities:
                    all_correct += 1
        
            for predict, predict_rod in zip(predict_entities, predict_entities_rod):
                if predict_rod == 'o':
                    overlap_pred += 1
                    if predict in gold_entities:
                        overlap_correct += 1

                if predict_rod == 'r':
                    regular_pred += 1
                    if predict in gold_entities:
                        regular_correct += 1



            for predict, predict_rod in zip(predict_entities, predict_entities_rod):
                if predict_rod == 'r':
                    if predict not in gold_entities:
                        print("regular FP: {}".format(predict))
                elif predict_rod == 'o':
                    if predict not in gold_entities:
                        print("overlap FP: {}".format(predict))


            for gold, gold_rod in zip(gold_entities, gold_entities_rod):
                if gold_rod == 'r':
                    if gold not in predict_entities:
                        print("regular FN: {}".format(gold))
                elif gold_rod == 'o':
                    if gold not in predict_entities:
                        print("overlap FN: {}".format(gold))

            offset += len(sentence)


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