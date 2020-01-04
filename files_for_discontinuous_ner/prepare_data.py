import os
from files_for_discontinuous_ner.my_corenlp_wrapper import StanfordCoreNLP
import json
from files_for_discontinuous_ner.data_structure import *

clef2013_train_src_dir = '/Users/feili/old_file/clef/2013/clef2013/task1train/ALLREPORTS'
clef2013_train_ann_dir = '/Users/feili/old_file/clef/2013/clef2013/task1train/CLEFPIPEDELIMITED-NoDuplicates'
clef2013_test_dir = '/Users/feili/old_file/clef/2013/clef2013/task1test'

nlp_tool = StanfordCoreNLP('http://localhost:{0}'.format(9000))

def get_stanford_annotations(text, core_nlp, port=9000, annotators='tokenize,ssplit,pos,lemma'):
    output = core_nlp.annotate(text, properties={
        "timeout": "10000",
        "ssplit.newlineIsSentenceBreak": "two",
        'annotators': annotators,
        'outputFormat': 'json'
    })
    if type(output) is str:
        output = json.loads(output, strict=False)
    return output

def get_sentences_and_tokens_from_stanford(document):
    stanford_output = get_stanford_annotations(document.text, nlp_tool)
    document.sentences = []
    for s_sent_idx, s_sent in enumerate(stanford_output['sentences']):
        sentence = Sentence()
        sentence.idx = s_sent_idx
        sentence.tokens = []
        for s_token_idx, stanford_token in enumerate(s_sent['tokens']):
            token = Token()
            token.idx = s_token_idx
            token.start = int(stanford_token['characterOffsetBegin'])
            token.end = int(stanford_token['characterOffsetEnd'])
            token.text = document.text[token.start:token.end]
            if len(token.text.strip()) == 0:
                print("WARNING: the text of the token are all space characters, ignore")
                continue
            # Make sure that the token text does not contain any space
            if len(token.text.split()) != 1:
                print("WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(token.text, token.text.replace(' ', '-')))
                token.text = token.text.replace(' ', '-')

            sentence.tokens.append(token)

        sentence.start = sentence.tokens[0].start
        sentence.end = sentence.tokens[-1].end
        sentence.text = document.text[sentence.start:sentence.end]

        document.sentences.append(sentence)

    for entity in document.entities:

        sent_idx_spans = []
        for span_start, span_end in entity.spans:
            entity.name.append(document.text[span_start:span_end])

            for sentence in document.sentences:
                if span_start >= sentence.start and span_end <= sentence.end:
                    sent_idx_spans.append(sentence.idx)
                    break

        if len(set(sent_idx_spans)) < 1 or len(set(sent_idx_spans)) > 1:
            raise RuntimeError("entity not in one sentence")

        entity.sent_idx = sent_idx_spans[0]
        sentence = document.sentences[sent_idx_spans[0]]

        for span_start, span_end in entity.spans:
            tkStart = -1
            tkEnd = -1
            for token in sentence.tokens:
                if token.start == span_start:
                    tkStart = token.idx
                if token.end == span_end:
                    tkEnd = token.idx

            if tkStart != -1 and tkEnd != -1:
                entity.tkSpans.append((tkStart, tkEnd))
            else:
                raise RuntimeError("tkStart == -1 or tkEnd == -1")

    return


def get_entities(ann_file):
    entities = []
    with open(ann_file, 'r') as fp:
        for line in fp:
            line = line.strip()
            if len(line) != 0:
                columns = line.split('||')
                entity = Entity()
                entity.type = columns[1]
                entity.norm_ids.append(columns[2])

                spanNumber = len(columns) - 3
                spanIdx = 0
                while spanIdx < spanNumber / 2:
                    entity.spans.append((int(columns[spanIdx * 2 + 3]), int(columns[spanIdx * 2 + 1 + 3])))
                    spanIdx += 1

                entities.append(entity)

    return entities




def loadClef2013(text_dir_path, ann_dir_path):

    documents = []

    for filename in os.listdir(text_dir_path):
        if filename.find('.txt') == -1:
            continue
        if filename.find('.pipe.txt') != -1:
            continue

        document = Document()
        document.name = filename

        text_file = os.path.join(text_dir_path, filename)
        ann_file = os.path.join(ann_dir_path, filename[:filename.find('.txt')]+'.pipe.txt')

        document.entities = get_entities(ann_file)

        with open(text_file, 'r') as fp:
            document.text = fp.read()

        get_sentences_and_tokens_from_stanford(document)

        documents.append(document)

    return documents



if __name__ == "__main__":

    documents = loadClef2013(clef2013_train_src_dir, clef2013_train_ann_dir)

    pass
