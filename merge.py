import os
import json

ace_data_dir = './data/ace05/processed-data/json-tf'
clef_data_dir = './data/clef/processed-data/json-tf'
merge_data_dir = './data/merge/'

def merge_ace_and_clef():

    for file_name in ['train.json', 'dev.json', 'test.json']:
        ace_fp = open(os.path.join(ace_data_dir, file_name), 'r')
        clef_fp = open(os.path.join(clef_data_dir, file_name), 'r')
        merge_fp = open(os.path.join(merge_data_dir, file_name), 'w')

        ace_lines = ace_fp.readlines() # one ace line denotes a document
        clef_lines = clef_fp.readlines() # one clef line denotes a sentence
        # so we make one merge line denotes a sentence

        for line in ace_lines:
            line = line.strip()
            if len(line) == 0:
                continue
            document = json.loads(line)

            for sent_idx, (sentence, ner, tree, tf) in enumerate(zip(document['sentences'], document['ner'], document['trees'], document['tf'])):

                merge_sentence = {}
                merge_sentence['doc_key'] = document['doc_key']+"_"+str(sent_idx)
                merge_sentence['sentences'] = [sentence]
                for entity in ner:
                    entity[2] = 'ET'
                merge_sentence['ner'] = [ner]
                merge_sentence['relations'] = [[]]
                merge_sentence['trees'] = [tree]
                if 'F6' in tf:
                    tf.pop('F6')
                if 'F7' in tf:
                    tf.pop('F7')
                merge_sentence['tf'] = [tf]
                merge_fp.write(json.dumps(merge_sentence)+'\n')


        for line in clef_lines:
            line = line.strip()
            if len(line) == 0:
                continue
            document = json.loads(line)
            for sent_idx, (sentence, ner, relation, tree, tf) in enumerate(zip(document['sentences'], document['ner'], document['relations'], document['trees'], document['tf'])):
                merge_sentence = {}
                merge_sentence['doc_key'] = document['doc_key'] + "_" + str(sent_idx)
                merge_sentence['sentences'] = [sentence]
                for entity in ner:
                    entity[2] = 'ET'
                merge_sentence['ner'] = [ner]
                merge_sentence['relations'] = [relation]
                merge_sentence['trees'] = [tree]
                if 'F6' in tf:
                    tf.pop('F6')
                if 'F7' in tf:
                    tf.pop('F7')
                merge_sentence['tf'] = [tf]
                merge_fp.write(json.dumps(merge_sentence) + '\n')

        ace_fp.close()
        clef_fp.close()
        merge_fp.close()

if __name__ == "__main__":

    merge_ace_and_clef()

    pass