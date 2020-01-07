import json
import shutil
import sys
import logging
import os
import sys

if os.environ.get("ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)


from allennlp.commands import main

experiment_name="clef"
data_root="./data/clef/processed-data/json"
config_file="./training_config/clef_working_example_debug.jsonnet"
cuda_device=-1


# ie_train_data_path=$data_root/train.json \
#     ie_dev_data_path=$data_root/dev.json \
#     ie_test_data_path=$data_root/test.json \
#     cuda_device=$cuda_device \
#     allennlp train $config_file \
#     --cache-directory $data_root/cached \
#     --serialization-dir ./models/$experiment_name \
#     --include-package dygie

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "--cache-directory", data_root+"/cached",
    "--serialization-dir", "./models/{}".format(experiment_name),
    "--include-package", 'dygie',
    '-f'
]

main()

# import os
# os.system('./scripts/train/train_scierc.sh -1')
