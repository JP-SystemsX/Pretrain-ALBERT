import sys

import datetime

import numpy as np
from datasets import load_metric
from transformers import AlbertTokenizerFast, DataCollatorForLanguageModeling
from transformers import AlbertForPreTraining, AlbertConfig
from transformers import Trainer, TrainingArguments
import json
import argparse

from sop_dataset import LineByLineWithSOPTextDataset

def parse_args():
    parser = argparse.ArgumentParser(description='''This Script is used to Pretrain A Light BERT''')

    parser.add_argument('--data_dir', type=str, required=True,
                        help='''
                            Path to the root dir of the tokenized dataset (above train and eval)
                            which contains the dataset which was created with the create_dataset.py script. 
                            ''',
                        default='/Datasets/RawTexts_tokenized')

    parser.add_argument('--model_path', type=str, required=False,
                        help='''The Model (with its corresponding Tokenizer) that shall be used.''',
                        default='albert-base-v2')

    parser.add_argument('--output_dir', type=str, required=False,
                        help='''
                        The script will output a pretrained ALBERT model to this address.
                        ''',
                        default='models/pretrain/')
    parser.add_argument('--use_comet', action='store_true', help='''
                        Add this flag to log your experiment to the comet.ml dashboard
                        ''')

    parser.add_argument('--from_scratch', action='store_true', help='''
                            Add this flag to use randomly initialized weights instead of further
                            pretraining the given model. 
                            ''')

    return parser.parse_args()

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions_mlm = np.argmax(logits[0], axis=-1)
    labels_filter_mlm = labels[0] != -100
    acc_mlm = metric.compute(predictions=predictions_mlm[labels_filter_mlm],
                             references=labels[0][labels_filter_mlm])

    predictions_sop = np.argmax(logits[1], axis=-1)
    acc_sop = metric.compute(predictions=predictions_sop, references=labels[1])

    return {'acc_mlm': acc_mlm['accuracy'], 'acc_sop': acc_sop['accuracy']}

if __name__ == '__main__':
    args = parse_args()
    if args.use_comet:
        import comet_ml

    tokenized_data_dir = args.data_dir

    dataset = LineByLineWithSOPTextDataset

    model_path = args.model_path  # huggingface model path, e.g. 'albert-base-v2'
    from_scratch = args.from_scratch

    tokenizer_info = json.load(open(tokenized_data_dir + '/train/info.json'))
    tokenizer_path = tokenizer_info['tokenizer_path']

    experiment_start = datetime.datetime.now().strftime("%y-%m-%d_%H:%M:%S:%f")
    out_dir = args.output_dir + f"{args.model_name}_{experiment_start}"
    print('Output dir:', out_dir)

    tokenizer = AlbertTokenizerFast.from_pretrained(tokenizer_path)
    metric = load_metric("accuracy")


    dataset_sop = dataset(
        load_from_path=tokenized_data_dir + '/train'
    )
    dataset_sop_eval = dataset(
        load_from_path=tokenized_data_dir + '/eval'
    )

    dataset_sop_eval.examples = dataset_sop_eval.examples[:500]
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=13,
        per_gpu_train_batch_size=16,
        per_device_eval_batch_size=2,
        save_total_limit=10,
        save_strategy='epoch',
        prediction_loss_only=False,
        evaluation_strategy="epoch",
        eval_accumulation_steps=1,
        label_names=['labels', 'sentence_order_label'],
        load_best_model_at_end=True  # according to eval_loss, if metric_for_best_model is not set
    )

    if from_scratch:
        # ALBERT base config: https://tfhub.dev/google/albert_base/1
        config = AlbertConfig.from_pretrained(model_path)
        model = AlbertForPreTraining(config)
    else:
        model = AlbertForPreTraining.from_pretrained(model_path)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_sop,
        compute_metrics=compute_metrics,
        eval_dataset=dataset_sop_eval
    )
    trainer.train()
    print(trainer.evaluate())

    if args.use_comet:
        experiment = comet_ml.config.get_global_experiment()
        experiment.log_parameters({
            'model_path': model_path,
            'tokenized_data_dir': tokenized_data_dir,
            'model_save_dir': out_dir,
            'experiment_start': experiment_start
        })
        experiment.log_parameters(tokenizer_info, prefix='dataset/')
        experiment.log_parameters(training_args.to_sanitized_dict(), prefix='train_args/')
        experiment.log_parameter('from_scratch', from_scratch)

    model.save_pretrained(out_dir, push_to_hub=False)
    print('Pre-Training done!')