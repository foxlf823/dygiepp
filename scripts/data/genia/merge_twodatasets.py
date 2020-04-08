
import json

# merge a and b into c
def merge_two_datasets(a, b, c):
    a_fp = open(a, 'r')
    b_fp = open(b, 'r')
    c_fp = open(c, "w")

    a_lines = a_fp.readlines()
    b_lines = b_fp.readlines()
    for a_line, b_line in zip(a_lines, b_lines):
        a_json = json.loads(a_line)
        b_json = json.loads(b_line)
        for k, v in b_json.items():
            if k not in a_json:
                a_json[k] = v
        c_fp.write(json.dumps(a_json)+"\n")

    a_fp.close()
    b_fp.close()
    c_fp.close()


if __name__ == '__main__':
    merge_two_datasets("./data/genia/processed-data/json-tf/train.json", "./data/genia/processed-data/json-dep/train.json",
                       "train.json")
    merge_two_datasets("./data/genia/processed-data/json-tf/dev.json",
                       "./data/genia/processed-data/json-dep/dev.json",
                       "dev.json")
    merge_two_datasets("./data/genia/processed-data/json-tf/test.json",
                       "./data/genia/processed-data/json-dep/test.json",
                       "test.json")