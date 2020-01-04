
from os import path
import os
import shutil

ace05_base = "./data/ace05"
ace05_split = f"{ace05_base}/ACE2005_split"
ace05_raw = f"{ace05_base}/English"
ace05_processed_raw = f"{ace05_base}/raw-data"
ace05_processed_json = f"{ace05_base}/processed-data"

def stat_dataset():
    stats = dict(train=0, dev=0, test=0)
    all_file_name = {}
    for fold in ["train", "dev", "test"]:
        fold_file = path.join(ace05_split, fold+".txt")
        lines = []
        with open(fold_file, 'r') as fp:
            for line in fp:
                line = line.strip()
                if len(line) == 0:
                    continue
                lines.append(line[line.rfind('/')+1:])
        stats[fold] = len(lines)
        all_file_name[fold] = lines

    print("file stat based on Lu's split")
    print(stats)

    print("find Lu's files in ACE05 dataset and copy them into {}".format(ace05_processed_raw))
    if not path.exists(ace05_processed_raw):
        os.mkdir(ace05_processed_raw)
    # sgm_dir = ace05_processed_raw + "/sgm"
    # apf_dir = ace05_processed_raw + "/apf"
    for fold in ["train", "dev", "test"]:
        # sgm_dir_fold = sgm_dir+"/"+fold
        # if not path.exists(sgm_dir_fold):
        #     os.makedirs(sgm_dir_fold)
        # apf_dir_fold = apf_dir+"/"+fold
        # if not path.exists(apf_dir_fold):
        #     os.makedirs(apf_dir_fold)
        if not path.exists(ace05_processed_raw+"/"+fold):
            os.makedirs(ace05_processed_raw+"/"+fold)

    stats = dict(match=0, unmatch=0)
    for folder in ['bc', 'bn', 'nw', 'wl']:
        folder_path = path.join(ace05_raw, folder, 'timex2norm')

        for file_name in os.listdir(folder_path):
            if file_name.find(".apf.xml") == -1:
                continue

            file_name = file_name[:file_name.find(".apf.xml")]
            b_match = False
            for fold in ["train", "dev", "test"]:
                fold_file_name  = all_file_name[fold]
                if file_name in fold_file_name:
                    b_match = True
                    break
            if b_match:
                stats['match'] += 1
                # shutil.copy(folder_path+'/'+file_name+".sgm", sgm_dir+"/"+fold+"/"+file_name+".sgm")
                # shutil.copy(folder_path + '/' + file_name + ".apf.xml", apf_dir + "/" +fold+"/"+ file_name + ".apf.xml")
                shutil.copy(folder_path + '/' + file_name + ".apf.xml",
                            ace05_processed_raw + "/" + fold + "/" + file_name + ".apf.xml")
                shutil.copy(folder_path + '/' + file_name + ".sgm", ace05_processed_raw + "/" + fold + "/" + file_name + ".sgm")
            else:
                stats['unmatch'] += 1

    print(stats)






def main():



    if not path.exists(ace05_processed_json):
        os.mkdir(ace05_processed_json)

    # stat_dataset()

if __name__ == '__main__':
    main()