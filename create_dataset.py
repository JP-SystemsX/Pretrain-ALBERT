from sop_dataset import LineByLineWithSOPTextDataset
from transformers import AlbertTokenizerFast
from os import makedirs
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='''
        This Script is used to prepare the data for pretraining A Light BERT model.
    ''')

    parser.add_argument('--data_dir', type=str, required=True,
                        help='''
                            Path to a directory which potentially can contain multiple files.
                            All these files will be processed as input data.
                            ''',
                        default='/Datasets/RawTexts')

    parser.add_argument('--model_name', type=str, required=False,
                        help='''The Model (with its corresponding Tokenizer) that shall be used.''',
                        default='albert-base-v2')

    parser.add_argument('--output_dir', type=str, required=False,
                        help='''
                        The script will output two datasets, one validation set and one training set to this address.
                        ''',
                        default='')

    return parser.parse_args()

args = parse_args()
dataset = LineByLineWithSOPTextDataset
model_path = args.model_name  # huggingface model path, e.g. 'albert-base-v2'
tokenizer = AlbertTokenizerFast.from_pretrained(model_path)

input_data_dir = args.data_dir
dataset_path = args.output_dir if len(args.output_dir) > 1 else input_data_dir + '_tokenized'

makedirs(dataset_path, exist_ok=True)


tokenized_data = dataset(
    tokenizer=tokenizer,
    file_dir=input_data_dir,
    block_size=512,
)

tokenized_data.save_train_eval_splits(dataset_path, eval_p=0.05)