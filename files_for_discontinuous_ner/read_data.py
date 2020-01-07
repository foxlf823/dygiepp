
import json

train_file = '/Users/feili/eclipse-workspace/learning-to-recognize-discontiguous-entities/train.txt'
dev_file = '/Users/feili/eclipse-workspace/learning-to-recognize-discontiguous-entities/dev.txt'
test_file = '/Users/feili/eclipse-workspace/learning-to-recognize-discontiguous-entities/test.txt'

output_dir = '/Users/feili/PycharmProjects/dygiepp/data/clef/processed-data/json'

# ALL_INSTANCES,
# ALL_WITH_ENTITIES,
# ONLY_CONTIGUOUS,
# CONTAIN_DISCONTIGUOUS,
# NO_DISCONTIGUOUS,
instanceFilter = 'CONTAIN_DISCONTIGUOUS'

def hasDiscontiguousEntity(instance):
    entities = instance['entities']
    for entity in entities:
        if len(entity['span']) > 1:
            return True
    return False


def read_file(file_path, instanceFilter):
    filtered_instances = []
    fp = open(file_path, 'r', encoding='utf-8')
    for line in fp:
        line = line.strip()
        if len(line) == 0:
            continue

        instance = json.loads(line)
        if instanceFilter == 'CONTAIN_DISCONTIGUOUS':
            if not hasDiscontiguousEntity(instance):
                continue
        else:
            pass

        filtered_instances.append(instance)

    fp.close()
    return filtered_instances

def do_statistics(instances):
    entity_1seg = 0
    entity_2seg = 0
    entity_3seg = 0
    stats = dict(span=0, span_common=0)
    for instance in instances:
        for entity in instance['entities']:
            if len(entity['span']) == 1:
                entity_1seg += 1
            elif len(entity['span']) == 2:
                entity_2seg += 1
            else:
                entity_3seg += 1

            for span in entity['span']:
                start_end = span.split(',')
                start = int(start_end[0])
                end = int(start_end[1])
                stats['span'] += 1
                if end-start+1 <= 6:
                    stats['span_common'] += 1

    print("sentence number: {}".format(len(instances)))
    print("1 segment entity: {}".format(entity_1seg))
    print("2 segment entity: {}".format(entity_2seg))
    print("3 segment entity: {}".format(entity_3seg))
    print(stats)

import os
def transfer_into_dygie(instances, output_file):

    fp = open(output_file, 'w')
    for idx, instance in enumerate(instances):
        doc = {}
        doc['doc_key'] = instance['doc']+"_"+str(instance['start'])+"_"+str(instance['end'])
        doc['sentences'] = []
        doc['sentences'].append(instance['tokens'])
        doc['ner'] = []
        ner_for_this_sentence = []
        doc['relations'] = []
        relation_for_this_sentence = []
        for entity in instance['entities']:
            for span in entity['span']:
                start = int(span.split(',')[0])
                end = int(span.split(',')[1])
                entity_output = [start, end, entity['type']]
                ner_for_this_sentence.append(entity_output)

            n_spans = len(entity['span'])
            candidate_indices = [(i, j) for i in range(n_spans) for j in range(n_spans) if i!=j] # undirected relations
            for i, j in candidate_indices:
                arg1_start = int(entity['span'][i].split(',')[0])
                arg1_end = int(entity['span'][i].split(',')[1])
                arg2_start = int(entity['span'][j].split(',')[0])
                arg2_end = int(entity['span'][j].split(',')[1])
                relation_for_this_sentence.append([arg1_start, arg1_end, arg2_start, arg2_end, "Combined"])

        doc['ner'].append(ner_for_this_sentence)
        doc['relations'].append(relation_for_this_sentence)
        fp.write(json.dumps(doc)+"\n")

    fp.close()

if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filtered_instances = read_file(train_file, instanceFilter)
    do_statistics(filtered_instances)
    transfer_into_dygie(filtered_instances, os.path.join(output_dir, 'train.json'))


    filtered_instances = read_file(dev_file, instanceFilter)
    do_statistics(filtered_instances)
    transfer_into_dygie(filtered_instances, os.path.join(output_dir, 'dev.json'))

    filtered_instances = read_file(test_file, instanceFilter)
    do_statistics(filtered_instances)
    transfer_into_dygie(filtered_instances, os.path.join(output_dir, 'test.json'))

    pass