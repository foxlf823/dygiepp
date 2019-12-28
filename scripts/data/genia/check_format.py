
import os
from os import path
import shutil
import re

#95363073, hetero dimers -> heterodimers, 42,47 G#protein -> 42,46 G#protein
#95329724, 'A pediatric oncology group study .' is a sentence
#95329723, EBNA 1 -> EBNA1, 12,25 G#protein -> 12,24 G#protein
#95238477, 'Several other transcription' is a sentence

# Notes
# 95256242 contain empty tokens 'CD4 - CD8 -  murine T', but they don't have postags
# 99228859 contain whitespace token, they have postags 'compounds ( 7 ,   11 )'
# postags can't correspond to tokens accurately, see 99410810 "( PPARgamma )"
# Therefore, we just do the following checking
# 1. remove empty or whitespace tokens (may lead to inaccurate entity start and end offsets)
# 2. tokenize some tokens such as 5' or FY*B
# 3. adjust entity start and end offsets based on tokenization
# 4. keep postags unchanged


def check_one_sentence(one_sentence):
    # assert(len(one_sentence['tokens']) == len(one_sentence['postags']))
    new_tokens = []
    new_postags = []
    # for idx, (token, postag) in enumerate(zip(one_sentence['tokens'], one_sentence['postags'])):
    for idx, token in enumerate(one_sentence['tokens']):
        if token == '':
            raise RuntimeError("token can't be empty")

            # for entity in one_sentence['entities']:
            #     if entity['start'] < idx:
            #         pass
            #     elif entity['start'] == idx:
            #         raise RuntimeError("entity['start'] can't be empty token")
            #     else:
            #         entity['start'] = entity['start']-1
            #
            #     if entity['end'] < idx+1:
            #         pass
            #     elif entity['end'] == idx+1:
            #         raise RuntimeError("entity['end'] can't be empty token")
            #     else:
            #         entity['end'] = entity['end']-1
                
        else:
            sub_tokens_ = re.split(r'([\'\*\(\)])', token)
            sub_tokens = [t for t in sub_tokens_ if t != '']
            m = len(sub_tokens)
            if m == 1:
                new_tokens.append(token)
                # new_postags.append(postag)
            elif m > 1:
                for sub_token in sub_tokens:
                    new_tokens.append(sub_token)
                    # new_postags.append(postag)
                for entity in one_sentence['entities']:
                    if entity['start'] < idx:
                        pass
                    elif entity['start'] == idx:
                        pass
                    else:
                        entity['start'] = entity['start'] + (m-1)

                    if entity['end'] < idx+1:
                        pass
                    elif entity['end'] == idx+1:
                        entity['end'] = entity['end'] + (m - 1)
                    else:
                        entity['end'] = entity['end'] + (m-1)
            else:
                raise RuntimeError("token can't be empty")

    one_sentence['tokens'] = new_tokens
    # one_sentence['postags'] = new_postags
    return one_sentence

def line_to_entities(line):
    entities = []
    for str_entity in line.split('|'):
        entity = dict()
        str_span, str_type = str_entity.split()
        entity['type'] = str_type
        entity['start'] = int(str_span.split(',')[0])
        entity['end'] = int(str_span.split(',')[1])
        entities.append(entity)
    return entities

def entities_to_line(entities):
    ret = ""
    for idx, entity in enumerate(entities):
        if idx == len(entities)-1:
            ret += str(entity['start'])+","+str(entity['end'])+" "+entity['type']
        else:
            ret += str(entity['start']) + "," + str(entity['end']) + " " + entity['type']+"|"
    return ret


# def split_by_one_space(line):
#     ret = []
#     tmp = ''
#     for ch in line:
#         if ch == ' ':
#             ret.append(tmp)
#             tmp = ''
#         else:
#             tmp += ch


def check_one_file(in_lines):
    out_lines = []
    one_sentence = dict()
    status = 0
    for line in in_lines:
        line = line.strip()
        if len(line) == 0 and status == 0:
            one_sentence = check_one_sentence(one_sentence)
            out_lines.append(' '.join(one_sentence['tokens'])+"\n")
            out_lines.append(' '.join(one_sentence['postags'])+"\n")
            out_lines.append(entities_to_line(one_sentence['entities'])+"\n")
            out_lines.append('\n')
        elif len(line) == 0 and status == 2:
            one_sentence['entities'] = []
            status = (status + 1) % 3
        else:
            if status == 0:
                one_sentence['tokens'] = line.split()
                status = (status + 1) % 3
            elif status == 1:
                one_sentence['postags'] = line.split()
                status = (status + 1) % 3
            elif status == 2:
                one_sentence['entities'] = line_to_entities(line)
                status = (status + 1) % 3

    return out_lines



def check_fold(fold, in_dir, out_dir):
    out_dir = path.join(out_dir, fold)
    in_dir = path.join(in_dir, fold)
    for file in os.listdir(in_dir):
        if file.find(".data") == -1:
            continue

        # if file.find('95256242') != -1:
        #     a = 2
        if file.find('95221386') != -1:
            a = 2
        in_fp = open(os.path.join(in_dir, file), "r")
        out_fp = open(os.path.join(out_dir, file), "w")
        in_lines = in_fp.readlines()
        out_lines = check_one_file(in_lines)
        out_fp.writelines(out_lines)
        in_fp.close()
        out_fp.close()


def main():
    in_prefix = "./data/genia/raw-data/sutd-article"
    in_dir = f"{in_prefix}/split-corrected"
    out_dir = f"{in_prefix}/split-corrected-checked"
    os.makedirs(out_dir)
    folds = ["train", "dev", "test"]
    for fold in folds:
        shutil.copy(path.join(in_dir, "{0}_order.csv".format(fold)), path.join(out_dir, "{0}_order.csv".format(fold)))

    for fold in folds:
        os.makedirs(f"{in_prefix}/split-corrected-checked/{fold}")
        msg = "Check fold {0}.".format(fold)
        print(msg)
        check_fold(fold, in_dir, out_dir)





if __name__ == "__main__":
    main()
