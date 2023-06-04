from torch.optim import AdamW
from utils import load
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import random
import tqdm
from tree import Tree
import argparse
import os
from typing import List


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset', type=str, default='../dataset/', help='path to dataset')
    parser.add_argument(
        '--path_to_save_trained_model', type=str, default='../model/', help='path to save trained model')
    parser.add_argument('--device', type=torch.device, default=torch.device('cuda'), help='device to use for training')

    args = parser.parse_args()
    return args


class TreeDataset(torch.utils.data.Dataset):
    def __init__(self, tree_list: List[Tree]) -> None:
        """class for tree dataset

        Parameters
        ----------
        tree_list : List[Tree]
            list of trees
        """
        self.tree_list = tree_list
        filtered_tree_list = []
        for tree in tree_list:
            if len(tree.sentence) > 30 or len(tree.sentence) < 10:
                continue
            filtered_tree_list.append(tree)
        self.tree_list = filtered_tree_list

    def __len__(self):
        return len(self.tree_list)

    def __getitem__(self, idx):
        return self.tree_list[idx]


tokenizer = RobertaTokenizer.from_pretrained('roberta-large')


def collate_fn(batch, tokenizer=tokenizer):
    input_ids = []
    labels = []
    for tree in batch:
        while True:
            node = random.choice(tree.node_list)
            if node.is_leaf or node.parent_node is None:
                continue
            elif len(node.content) < 2 or len(node.content) > 6:
                continue
            else:
                break
        sentence = " ".join(tree.sentence)
        phrase = " ".join(node.content)
        # devide phrase into subwords
        subwords = tokenizer.tokenize(phrase)
        num_subwords = len(tokenizer.tokenize(phrase))
        label = tokenizer.encode(sentence, return_tensors='pt')[0]
        # replace part of phrase with mask tokens seprarated by space
        for i in range(num_subwords):
            mask_tokens = ''.join([tokenizer.mask_token] * (num_subwords - i))
            part_of_phrase = tokenizer.convert_tokens_to_string(subwords[:i])
            masked_sentence = sentence.replace(phrase, part_of_phrase + mask_tokens)
            input_id = tokenizer.encode(masked_sentence, return_tensors='pt')[0]
            if input_id.shape[0] == label.shape[0]:
                input_ids.append(input_id)
                labels.append(torch.where(input_id == tokenizer.mask_token_id, label, -100))
    # pad input_ids and labels
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, padding_value=tokenizer.pad_token_id, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(labels, padding_value=tokenizer.pad_token_id, batch_first=True)

    # make attention mask
    attention_mask = torch.where(input_ids == tokenizer.pad_token_id, 0, 1)
    return input_ids, labels, attention_mask


def train():
    args = arg_parse()

    # build model
    model = RobertaForMaskedLM.from_pretrained('roberta-large')
    model.to(args.device)

    # build train dataset
    train_tree_list = load(os.path.join(args.path_to_dataset, 'tree_list/train_tree_list.pickle'))
    train_dataset = TreeDataset(train_tree_list.tree_list)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # build dev dataset
    dev_tree_list = load(os.path.join(args.path_to_dataset, 'tree_list/dev_tree_list.pickle'))
    dev_dataset = TreeDataset(dev_tree_list.tree_list)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # define AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # for early stop
    max_acc = 0
    patience = 3
    for epoch in range(50):
        with tqdm.tqdm(total=len(train_dataloader)) as pbar:
            pbar.set_description(f'Epoch {epoch}:')
            for input_ids, label, attention_mask in train_dataloader:
                input_ids = input_ids.to(args.device)
                label = label.to(args.device)
                attention_mask = attention_mask.to(args.device)
                optimizer.zero_grad()
                outputs = model(input_ids, labels=label, attention_mask=attention_mask)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                # caluculate accuracy
                pred = torch.argmax(outputs.logits, dim=-1)
                acc = torch.where(pred == label, 1, 0)
                acc = torch.where(label == -100, 0, acc)
                acc = torch.sum(acc) / torch.sum(label != -100)
                pbar.set_postfix(loss=loss.item(), acc=acc.item())
                pbar.update(1)

        with tqdm.tqdm(total=len(dev_dataloader)) as pbar:
            pbar.set_description('Evaluating:')
            val_loss = 0
            val_acc = 0
            for input_ids, label, attention_mask in dev_dataloader:
                input_ids = input_ids.to(args.device)
                label = label.to(args.device)
                attention_mask = attention_mask.to(args.device)
                outputs = model(input_ids, labels=label, attention_mask=attention_mask)
                loss = outputs.loss
                pred = torch.argmax(outputs.logits, dim=-1)
                acc = torch.where(pred == label, 1, 0)
                acc = torch.where(label == -100, 0, acc)
                acc = torch.sum(acc) / torch.sum(label != -100)
                val_loss += loss.item()
                val_acc += acc.item()
                pbar.set_postfix(loss=loss.item(), acc=acc.item())
                pbar.update(1)
            val_loss /= len(dev_dataloader)
            val_acc /= len(dev_dataloader)

        # check the val_acc for early sotpping of training
        if val_acc > max_acc:
            max_acc = val_acc
            patience = 3
            # save model
            torch.save(model.state_dict(), os.path.join(args.path_to_save_trained_model, 'infilling_roberta.pth'))
        else:
            patience -= 1
            if patience == 0:
                print(f'Early stop at epoch {epoch}')
                break


if __name__ == '__main__':
    train()
