import os
import shutil

semeval2015task14_development_data = '/Users/feili/old_file/sem2015 task 14/semeval-2015-task-14-devel/data/devel/discharge'
test_file_list = '/Users/feili/eclipse-workspace/learning-to-recognize-discontiguous-entities/data/test.filelist'
output_dir = './ShAReCLEFeHealth2014'

def check():
    test_file_names = []
    with open(test_file_list, 'r') as fp:
        for line in fp:
            line = line.strip()
            if len(line) == 0:
                continue
            test_file_names.append(line[:line.find('-DISCHARGE_SUMMARY.txt')])

    stat = dict(match=0, unmatch=0)
    for file_name in os.listdir(semeval2015task14_development_data):
        if file_name.find('.text') != -1:
            short_name = file_name[:file_name.find('.text')]
            if short_name in test_file_names:
                stat['match'] += 1
            else:
                stat['unmatch'] += 1
    print(stat)

def prepare_text_files():

    if not os.path.exists(os.path.join(output_dir, 'text')):
        os.makedirs(os.path.join(output_dir, 'text'))
    for file_name in os.listdir(semeval2015task14_development_data):
        if file_name.find('.text') != -1:
            short_name = file_name[:file_name.find('.text')]
            shutil.copy(os.path.join(semeval2015task14_development_data, file_name), os.path.join(output_dir, 'text', short_name+"-DISCHARGE_SUMMARY.txt"))

def prepare_annotation_files():
    if not os.path.exists(os.path.join(output_dir, 'annotation')):
        os.makedirs(os.path.join(output_dir, 'annotation'))

    for file_name in os.listdir(semeval2015task14_development_data):
        if file_name.find('.pipe') != -1:
            short_name = file_name[:file_name.find('.pipe')]

            existing_mention = []
            with open(os.path.join(semeval2015task14_development_data, file_name), 'r') as fp:
                with open(os.path.join(output_dir, 'annotation', short_name+"-DISCHARGE_SUMMARY.pipe.txt"), 'w') as fp_out:
                    for line in fp:
                        line = line.strip()
                        if len(line) == 0:
                            continue

                        columns = line.split("|")
                        if columns[1] in existing_mention:
                            continue
                        existing_mention.append(columns[1])
                        clef_line = []
                        clef_line.append(columns[0][:columns[0].find('.text')]+"-DISCHARGE_SUMMARY.txt")
                        clef_line.append('Disease_Disorder')
                        clef_line.append(columns[2])
                        for sub1 in columns[1].split(','):
                            for sub2 in sub1.split('-'):
                                clef_line.append(sub2)

                        fp_out.write('||'.join(clef_line)+"\n")





if __name__ == "__main__":
    # check whether semeval2015task14_development_data match test_file_list
    # check()

    # prepare text file to output_dir
    # prepare_text_files()

    # prepare annotation file to output_dir, need to transfer the format from semeval to clef
    prepare_annotation_files()

    pass