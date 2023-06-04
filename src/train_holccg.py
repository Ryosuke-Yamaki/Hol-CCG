import os
from utils import load
from evaluation_functions import evaluate_stag, evaluate_batch_list
import torch
import torch.nn as nn
import torch.optim as optim
from tree import Tree
from holccg import HolCCG
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
import wandb
import datetime
import argparse
from typing import List, Tuple


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_tree_list', type=str, default='../dataset/tree_list/', help='path to tree list')
    parser.add_argument(
        '--path_to_save_trained_model', type=str, default='../model/', help='path to save trained model')
    parser.add_argument(
        '--encoder',
        choices=[
            'bert-base-cased',
            'bert-large-cased',
            'roberta-base',
            'roberta-large'
        ],
        type=str,
        default='roberta-large', help='pretrained text encoder')
    parser.add_argument('--phrase_loss_weight', type=float, default=1.0, help='weight of phrase loss')
    parser.add_argument('--span_loss_weight', type=float, default=1.0, help='weight of span loss')
    parser.add_argument('--normalize', choices=['real', 'complex'], default='real', help='normalize type')
    parser.add_argument(
        '--composition',
        type=str,
        choices=[
            'corr',
            'conv',
            's_conv'],
        default='corr', help='composition type')
    parser.add_argument('--max_norm', type=float, default=30.0, help='max norm of vector')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='base learning rate')
    parser.add_argument('--ft_lr', type=float, default=1e-5, help='fine-tuning learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument(
        '--device',
        type=torch.device,
        default=torch.device('cuda'),
        help='device to use for training')
    parser.add_argument('--wandb', action='store_true', help='use wandb for logging')

    args = parser.parse_args()
    return args


def load_tree_list(path_to_tree_list: str) -> Tuple[List[Tree], List[Tree]]:
    """Load train and dev tree list.

    Parameters
    ----------
    path_to_tree_list : str
        path to tree list directory

    Returns
    -------
    Tuple[List[Tree], List[Tree]]
        train tree list and dev tree list
    """
    print('Loading tree list...')
    train_tree_list = load(os.path.join(path_to_tree_list, 'train_tree_list.pickle'))
    dev_tree_list = load(os.path.join(path_to_tree_list, 'dev_tree_list.pickle'))
    return train_tree_list, dev_tree_list


def build_encoder_and_tokenizer(encoder_name: str) -> Tuple[nn.Module, nn.Module, int]:
    """Build pretrained text encoder and tokenizer.

    Parameters
    ----------
    encoder_name : str
        name of pretrained text encoder

    Returns
    -------
    Tuple[nn.Module, nn.Module, int]
        pretrained text encoder, tokenizer, and model dimension
    """
    print(f"Loading pretrained {encoder_name} encoder...")
    if 'roberta' in encoder_name:
        encoder = RobertaModel.from_pretrained(encoder_name)
        tokenizer = RobertaTokenizer.from_pretrained(encoder_name)
    elif 'bert' in encoder_name:
        encoder = BertModel.from_pretrained(encoder_name)
        tokenizer = BertTokenizer.from_pretrained(encoder_name)

    if 'base' in encoder_name:
        model_dim = 768
    elif 'large' in encoder_name:
        model_dim = 1024
    return encoder, tokenizer, model_dim


def log_stat_to_wandb(stat: dict, prefix: str, epoch: int) -> None:
    """Log statistics to wandb.

    Parameters
    ----------
    stat : dict
        statistics
    prefix : str
        prefix of statistics
    epoch : int
        epoch
    """
    for key, value in stat.items():
        wandb.log({f'{prefix}_{key}': value}, step=epoch)


def train():
    args = arg_parse()
    if args.normalize == 'real':
        max_norm = args.max_norm
    elif args.normalize == 'complex':
        max_norm = None

    # convert the args to dict
    hyper_params = vars(args)

    trained_model_name = args.encoder + '_word'
    if args.phrase_loss_weight != 0.0:
        trained_model_name += '_phrase'
    if args.span_loss_weight != 0.0:
        trained_model_name += '_span'
    trained_model_name += '_' + str(datetime.datetime.now()).split('.')[0].replace(' ', '_')
    trained_model_name += '.pth'
    path_to_save_trained_model = os.path.join(args.path_to_save_trained_model, trained_model_name)

    encoder, tokenizer, model_dim = build_encoder_and_tokenizer(args.encoder)

    train_tree_list, dev_tree_list = load_tree_list(args.path_to_tree_list)

    # set info for training
    train_tree_list.device = args.device
    dev_tree_list.device = args.device
    with torch.no_grad():
        train_tree_list.set_info_for_training(tokenizer)
        dev_tree_list.set_info_for_training(tokenizer)

    # the number of word category and phrase category
    # these are used for building HolCCG's syntactic classifier
    num_word_cat = len(train_tree_list.word_category_vocab.stoi)
    num_phrase_cat = len(train_tree_list.phrase_category_vocab.stoi)

    # build HolCCG
    holccg = HolCCG(
        num_word_cat=num_word_cat,
        num_phrase_cat=num_phrase_cat,
        encoder=encoder,
        tokenizer=tokenizer,
        model_dim=model_dim,
        dropout=args.dropout,
        normalize_type=args.normalize,
        vector_norm=max_norm,
        composition=args.composition,
        device=args.device).to(args.device)
    holccg.eval()

    criteria = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([{'params': holccg.base_params},
                            {'params': holccg.encoder.parameters(), 'lr': args.ft_lr}], lr=args.base_lr)

    if args.wandb:
        wandb.init(project='Hol-CCG', name=trained_model_name.replace('.pth', ''), config=hyper_params)

    # evaluate initial state
    dev_batch_list = dev_tree_list.make_batch(args.batch_size)
    with torch.no_grad():
        dev_stat = evaluate_batch_list(dev_batch_list, holccg)
        dev_tree_list.set_vector(holccg)
        stag_acc = evaluate_stag(dev_tree_list, holccg)
    dev_stat['stag_acc'] = stag_acc
    if args.wandb:
        log_stat_to_wandb(dev_stat, 'dev', 0)

    for epoch in range(1, args.epochs + 1):
        holccg.train()
        train_batch_list = train_tree_list.make_batch(args.batch_size)
        epoch_word_loss = 0.0
        epoch_phrase_loss = 0.0
        epoch_span_loss = 0.0
        num_batch = 0
        with tqdm(total=len(train_batch_list), unit="batch") as pbar:
            pbar.set_description(f"Epoch[{epoch}/{args.epochs}]")
            for batch in train_batch_list:
                optimizer.zero_grad()
                word_output, phrase_output, span_output, word_label, phrase_label, span_label = holccg(
                    batch)
                word_loss = criteria(word_output, word_label)
                phrase_loss = criteria(phrase_output, phrase_label)
                span_loss = criteria(span_output, span_label)

                loss_to_backward = word_loss + args.phrase_loss_weight * phrase_loss + args.span_loss_weight * span_loss
                loss_to_backward.backward()
                optimizer.step()

                epoch_word_loss += word_loss.item()
                epoch_phrase_loss += phrase_loss.item()
                epoch_span_loss += span_loss.item()

                num_batch += 1
                pbar.set_postfix({"word-loss": epoch_word_loss / num_batch,
                                  "phrase-loss": epoch_phrase_loss / num_batch,
                                  "span-loss": epoch_span_loss / num_batch})
                pbar.update(1)

        holccg.eval()
        with torch.no_grad():
            dev_stat = evaluate_batch_list(dev_batch_list, holccg)
            dev_tree_list.set_vector(holccg)
            stag_acc = evaluate_stag(dev_tree_list, holccg)
        dev_stat['stag_acc'] = stag_acc
        torch.save(holccg, path_to_save_trained_model)
        if args.wandb:
            log_stat_to_wandb(dev_stat, 'dev', epoch)

    # load and prepare batch of test_tree_list
    test_tree_list = load(os.path.join(args.path_to_tree_list, 'test_tree_list.pickle'))
    test_tree_list.device = args.device
    test_tree_list.set_info_for_training(tokenizer)
    test_batch_list = test_tree_list.make_batch(args.batch_size)

    holccg = torch.load(path_to_save_trained_model, map_location=args.device)
    holccg.eval()

    with torch.no_grad():
        test_stat = evaluate_batch_list(test_batch_list, holccg)
        test_tree_list.set_vector(holccg)
        stag_acc = evaluate_stag(test_tree_list, holccg)
    test_stat['stag_acc'] = stag_acc

    if args.wandb:
        log_stat_to_wandb(test_stat, 'test', 0)


if __name__ == '__main__':
    train()
