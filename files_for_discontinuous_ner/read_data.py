
import json

train_file = '/Users/feili/eclipse-workspace/learning-to-recognize-discontiguous-entities/train.txt'
dev_file = '/Users/feili/eclipse-workspace/learning-to-recognize-discontiguous-entities/dev.txt'
test_file = '/Users/feili/eclipse-workspace/learning-to-recognize-discontiguous-entities/test.txt'

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

if __name__ == "__main__":
    filtered_instances = read_file(train_file, instanceFilter)

    do_statistics(filtered_instances)

    filtered_instances = read_file(dev_file, instanceFilter)

    do_statistics(filtered_instances)

    filtered_instances = read_file(test_file, instanceFilter)

    do_statistics(filtered_instances)

    pass