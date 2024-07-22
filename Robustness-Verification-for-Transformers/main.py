# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.
import argparse
import json
import os
import time

import psutil

# Make it possible to setup the process_range in software
import sys

from typing import List

from Parser import Parser, update_arguments
from transformers import AutoModelForSequenceClassification, BertTokenizer

argv = sys.argv[1:]
parser = Parser.get_parser()
args, _ = parser.parse_known_args(argv)

args = update_arguments(args)

# Setting up the GPU
if args.gpu != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Settin up the CPU
if psutil.cpu_count() > 4 and args.cpu_range != "Default":
    start, end = int(args.cpu_range.split("-")[0]), int(args.cpu_range.split("-")[1])
    os.sched_setaffinity(0, {i for i in range(start, end + 1)})

# from Verifiers.VerifierBackwardCombination import VerifierBackwardConvexCombination
import random
from Verifiers.attack import pgd_attack_bert
from Logger import Logger
from data_utils import load_data, get_batches, set_seeds, sample
from Models import Transformer
from Verifiers import VerifierForward, VerifierBackward, VerifierDiscrete, VerifierBackwardConvexCombination, \
    VerifierForwardConvexCombination, VerifierBackwardForwardConvexCombination, VerifierZonotope

# from eval_words import eval_words


# args.train = False
# args.verify = True
# args.cpus = 4
# args.data = "sst"
# args.base_dir = "bert-custom-word-vocab-uncased"
# args.base_dir = "bert-base-uncased"
# args.dir = "dev"
# args.method = "backward"
# args.cpu = True
# args.num_layers = 1

set_seeds(args.seed)

data_train, data_valid, data_test, _, _ = load_data(args)
#
# Add synonym data
if (args.verify and args.attack_type == "synonym" and args.data == "sst") or args.lirpa_data:
    with open('Verifiers/remplacements/dev.json') as valid_f, open('Verifiers/remplacements/test.json') as test_f:
        data_valid = json.load(valid_f)
        data_test = json.load(test_f)

        for data in [data_valid, data_test]:
            for example in data:
                tokens = example['sentence'].split()
                example['sent_a'] = tokens
                del example['sentence']

set_seeds(args.seed)


def build_lirpa_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--robust', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--dir', type=str, default='model')
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--data', type=str, default='sst', choices=['sst'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--legacy_loading', action='store_true',
                        help='use a deprecated way of loading checkpoints for previously saved models')
    parser.add_argument('--auto_test', action='store_true')

    parser.add_argument('--eps', type=float, default=1.0)
    parser.add_argument('--budget', type=int, default=6)
    parser.add_argument('--method', type=str, default=None,
                        choices=['IBP', 'IBP+backward', 'IBP+backward_train', 'forward', 'forward+backward'])

    parser.add_argument('--model', type=str, default='transformer',
                        choices=['transformer', 'lstm'])
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--num_epochs_all_nodes', type=int, default=20)
    parser.add_argument('--eps_start', type=int, default=1)
    parser.add_argument('--eps_length', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--min_word_freq', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--oracle_batch_size', type=int, default=1024)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_sent_length', type=int, default=32)
    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=1)
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--intermediate_size', type=int, default=128)
    parser.add_argument('--drop_unk', action='store_true')
    parser.add_argument('--hidden_act', type=str, default='relu')
    parser.add_argument('--layer_norm', type=str, default='no_var',
                        choices=['standard', 'no', 'no_var'])
    parser.add_argument('--loss_fusion', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--bound_opts_relu', type=str, default='zero-lb')

    return parser


synonyms_dict = None
if args.train and args.train_adversarial:
    with open('synonyms.json') as f:
        synonyms_dict = json.load(f)


def get_word_or_random_alternative(word):
    if word not in synonyms_dict:
        return word

    return random.choice([word] + synonyms_dict[word])


def modify_batch(batch: List):
    result = []
    for example in batch:
        new_sent_a = []
        for word in example['sent_a']:
            new_sent_a.append(get_word_or_random_alternative(word))

        result.append({"sent_a": new_sent_a, 'label': example['label']})

    return result


if args.attack_type == "synonym" and "adversarial" not in args.dir:
    args.with_lirpa_transformer = True
else:
    args.with_lirpa_transformer = False

# Load data
if args.attack_type == "synonym":
    from Models.TransformerLirpa.data_utils import load_data as load_data_lirpa
    data_train_all_nodes_lirpa, data_train_lirpa, data_dev_lirpa, data_test_lirpa = load_data_lirpa(args.data)
else:
    data_train_adversarial = data_train[:]
    if args.train_adversarial:
        for example in data_train:
            batch_adversarial = modify_batch([example, example, example, example, example])
            data_train_adversarial.extend(batch_adversarial)

    data_train = data_train_adversarial

# Load transformer
if args.with_lirpa_transformer:
    if args.lirpa_ckpt == "None":
        print("For synonym attack, must specifiy lirpa-ckpt arg (which indicates the checkpoint of the adversarially trained network)")
        exit(1)

    from Models.TransformerLirpa.Transformer import Transformer as TransformerLIRPA

    lirpa_parser = build_lirpa_parser()
    lirpa_args = lirpa_parser.parse_args([f"--load={args.lirpa_ckpt}"])

    lirpa_args.num_layers = args.num_layers
    if 'smaller' in args.lirpa_ckpt:
        lirpa_args.embedding_size = 64
        lirpa_args.hidden_size = 64
        lirpa_args.intermediate_size = 128
        args.num_input_error_terms = 64
    else:
        lirpa_args.embedding_size = 128
        lirpa_args.hidden_size = 128
        lirpa_args.intermediate_size = 128
        args.num_input_error_terms = 128

    target = TransformerLIRPA(lirpa_args, data_train_lirpa)
    target.eval()
else:
    target = Transformer(args, data_train)



import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, BertTokenizer
import random
from tqdm import tqdm

# 데이터셋을 DataLoader로 변환하는 함수
def create_data_loader(data, tokenizer, max_length, batch_size):
    inputs = []
    masks = []
    labels = []
    
    for example in data:
        encoded_dict = tokenizer.encode_plus(
            example['sent_a'],                      # 입력 텍스트
            add_special_tokens=True,                # [CLS]와 [SEP] 토큰 추가
            max_length=max_length,                  # 최대 길이
            padding='max_length',                   # 최대 길이로 패딩
            truncation=True,                        # 최대 길이로 자르기
            return_attention_mask=True,             # 어텐션 마스크 생성
            return_tensors='pt',                    # 텐서 반환
        )
        
        inputs.append(encoded_dict['input_ids'])
        masks.append(encoded_dict['attention_mask'])
        labels.append(example['label'])
    
    inputs = torch.cat(inputs, dim=0)
    masks = torch.cat(masks, dim=0)
    labels = torch.tensor(labels)
    
    dataset = TensorDataset(inputs, masks, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    return dataloader

# 모델과 토크나이저 로드
pretrained_model = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")
pretrained_state_dict = pretrained_model.state_dict()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# target 모델에도 동일한 사전 로드
if 'bert.embeddings.position_ids' in pretrained_state_dict:
    print("Removing position_ids from the state_dict")
    del pretrained_state_dict['bert.embeddings.position_ids']

try:
    target.model.load_state_dict(pretrained_state_dict, strict=True)
    print("Model parameters loaded successfully.")
except RuntimeError as e:
    print(f"Error loading model parameters: {e}")

# DataLoader 생성
batch_size = 32
max_length = 128

# 데이터셋에서 랜덤으로 10개의 예제 선택
random.seed(42)
random_samples = random.sample(data_test, 10)

dataloader = create_data_loader(random_samples, tokenizer, max_length, batch_size)

# 모델을 평가 모드로 전환
pretrained_model.eval()
target.model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model.to(device)
target.model.to(device)

# 데이터셋 분류 수행 및 출력
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Inference", unit="batch"):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs_pretrained = pretrained_model(input_ids, attention_mask=attention_mask)
        logits_pretrained = outputs_pretrained.logits
        predictions_pretrained = torch.argmax(logits_pretrained, dim=1)

        outputs_target = target.model(input_ids, attention_mask=attention_mask)
        logits_target = outputs_target[0]
        predictions_target = torch.argmax(logits_target, dim=1)

        # 각 예제에 대해 출력
        for i in range(input_ids.size(0)):
            # 원래 문장 출력
            original_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            print(f"Original Text: {original_text}")
            
            # pretrained_model 임베딩 출력
            embedding_output_pretrained = pretrained_model.bert.embeddings(input_ids[i].unsqueeze(0)).cpu().numpy()
            print(f"Pretrained Model Embedding Output:\n{embedding_output_pretrained}")
            
            # target 모델 임베딩 출력
            embedding_output_target = target.model.bert.embeddings(input_ids[i].unsqueeze(0)).cpu().numpy()
            print(f"Target Model Embedding Output:\n{embedding_output_target}")
            
            # 정답 레이블과 예측 레이블 출력
            print(f"True Label: {labels[i].item()}")
            print(f"Pretrained Model Predicted Label: {predictions_pretrained[i].item()}")
            print(f"Target Model Predicted Label: {predictions_target[i].item()}")
            print("="*50)
            if i == 0:
                break




# 프루닝 헤드 설정하는 부분
pruning_heads = [

]

# 모델의 모든 레이어에 대해 프루닝 수행
for i, layer in enumerate(target.model.bert.encoder.layer):
    if i < len(pruning_heads):
        layer.attention.prune_heads(pruning_heads[i])


print(args.cpu)
print(args.device)
print(args)

import tensorflow as tf

tf1 = tf.compat.v1

if tf.__version__[0] == '2':
    tf1.disable_eager_execution()

# config = tf.ConfigProto(device_count={'GPU': 0})
# config.gpu_options.allow_growth = True
sess = tf1.Session()  # config=config)





with sess.as_default():
    random.shuffle(data_valid)
    random.shuffle(data_test)
    valid_batches = get_batches(data_valid, args.batch_size)
    test_batches = get_batches(data_test, args.batch_size)
    print("Dataset sizes: %d/%d/%d" % (len(data_train), len(data_valid), len(data_test)))

    summary_names = ["loss", "accuracy"]
    summary_num_pre = 2

    logger = Logger(sess, args, summary_names, 1)

    print("\n")

    if args.train:
        # from Models.TransformerLirpa.data_utils import load_data as load_data_lirpa

        # data_train_all_nodes_lirpa, data_train_lirpa, data_dev_lirpa, data_test_lirpa = load_data_lirpa("sst")

        while logger.epoch.eval() <= args.num_epoches:
            random.shuffle(data_train)

            do_batch_size_test = False
            if do_batch_size_test:
                for batch_exp in range(5, 100):
                    size = 2 ** batch_exp
                    test_batches = get_batches(data_train, size)
                    example_batch = []
                    for x in test_batches[0]:
                        x['sentence'] = ' '.join(x['sent_a'])
                        example_batch.append(x)

                    start = time.time()

                    if args.with_lirpa_transformer:
                        p = target.forward(example_batch)
                    else:
                        p = target.step(test_batches[0], is_train=False)
                    end = time.time()
                    print(f"Processing batch with size {size} took {end - start} seconds")

            train_batches = get_batches(data_train, args.batch_size)
            for i, batch in enumerate(train_batches):
                logger.next_step(target.step(batch, is_train=True)[:summary_num_pre])
            target.save(logger.epoch.eval())
            logger.next_epoch()
            for batch in valid_batches:
                logger.add_valid(target.step(batch)[:summary_num_pre])
            logger.save_valid(log=True)
            for batch in test_batches:
                logger.add_test(target.step(batch)[:summary_num_pre])
            logger.save_test(log=True)

    if args.diffai:
        verifier = VerifierZonotope(args, target, logger)
        random.seed(args.seed)
        random.shuffle(data_train)
        train_batches = get_batches(data_train, args.batch_size)
        for i, example in enumerate(data_valid):
            verifier.train_diffai(example, eps=args.diffai_eps)

        for batch in valid_batches:
            logger.add_valid(target.step(batch)[:summary_num_pre])
        logger.save_valid(log=True)
        for batch in test_batches:
            logger.add_test(target.step(batch)[:summary_num_pre])
        logger.save_test(log=True)

        exit(0)

    data = data_valid if args.use_dev else data_test

    if args.pgd:
        examples = sample(args, data, target)
        pgd_attack_bert(target, args, examples)
        exit(0)

    elif args.debug_zonotope:
        verifier = VerifierZonotope(args, target, logger)
        random.seed(args.seed)
        zonotopes, example = verifier.collect_zonotopes_bounds(data, sentence_num=0, word_num=2, eps=args.max_eps)
        print(example)
        violations_per_layers = verifier.check_samples(example, zonotopes, sentence_num=0, word_num=2, eps=args.max_eps, num_samples=1000)
        print(violations_per_layers)
        print(example)
        exit(0)

    elif args.verify:
        print("Verifying robustness...")
        if args.method == "forward-convex":
            verifier = VerifierForwardConvexCombination(args, target, logger)
        elif args.method == "backward-convex":
            verifier = VerifierBackwardConvexCombination(args, target, logger)
        elif args.method == "baf-convex":
            verifier = VerifierBackwardForwardConvexCombination(args, target, logger)
        elif args.method == "forward" or args.method == "ibp":
            verifier = VerifierForward(args, target, logger)
        elif args.method == "backward" or args.method == "baf":
            verifier = VerifierBackward(args, target, logger)
        elif args.method == "discrete":
            verifier = VerifierDiscrete(args, target, logger)
        elif args.method == "zonotope":
            verifier = VerifierZonotope(args, target, logger)
        else:
            raise NotImplementedError("Method not implemented".format(args.method))

        if args.attack_type == "lp":
            verifier.run(data)
        elif args.attack_type == "synonym":
            # verifier.run_sentence_attacks(data_test_lirpa)
            # verifier.run_sentence_attacks(data_train_lirpa)
            # TODO: add back
            verifier.run_sentence_attacks(data)
        else:
            raise NotImplementedError()
        exit(0)

    # test the accuracy
    if args.lirpa_data:
        test_batches = get_batches(data_test_lirpa, args.batch_size)

    acc = 0
    for batch in test_batches:
        # import pdb; pdb.set_trace()
        if args.with_lirpa_transformer:
            preds = target.step(batch)[0]
            is_accurate = [x['label'] == preds[i] for i, x in enumerate(batch)]
            acc += sum(is_accurate).item()
        else:
            acc += target.step(batch)[1] * len(batch)
    acc = float(acc / len(data_test))
    print("Accuracy: {:.3f}".format(acc))
    with open(args.log, "w") as file:
        file.write("{:.3f}".format(acc))
