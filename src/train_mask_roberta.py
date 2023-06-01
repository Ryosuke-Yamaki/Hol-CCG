# import RoBERTa from transformers
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import random
import tqdm
import wandb

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

class Tree_Dataset(torch.utils.data.Dataset):
    def __init__(self, tree_list):
        self.tree_list = tree_list
        filtered_tree_list = []
        for tree in tree_list:
            if len(tree.sentence) > 30 or len(tree.sentence) < 10:
                continue
            filtered_tree_list.append(tree)
        self.tree_list = filtered_tree_list
        # self.set_info()               

    def __len__(self):
        return len(self.tree_list)

    def __getitem__(self, idx):
        return self.tree_list[idx]
    
    # def set_info(self):
    #     with tqdm.tqdm(total=len(self.tree_list)) as pbar:
    #         for tree in self.tree_list:
    #             sentence = " ".join(tree.sentence)
    #             tokenized_sentence = tokenizer.tokenize(sentence)
    #             tree.tokenized_sentence = tokenized_sentence
    #             for node in tree.node_list:
    #                 find = False
    #                 # check the start position and end position of node.content in tokenized sentence
    #                 for i in range(len(tokenized_sentence)):
    #                     for j in range(len(tokenized_sentence) - i):
    #                         subword = tokenizer.convert_tokens_to_string(tokenized_sentence[i:i+j])
    #                         # when find the start position and end position, set the node.tokenized_start_position and node.tokenized_end_position
    #                         if subword == " ".join(node.content):
    #                             node.tokenized_start_position = i
    #                             node.tokenized_end_position = i + j
    #                             find = True
    #                             break
    #                     if find:
    #                         break
                            
    #             pbar.update(1)


def collate_fn(batch, tokenizer=tokenizer):
    input_ids = []
    labels = []
    for tree in batch:
        # if len(tree.sentence) > 30 or len(tree.sentence) < 10:
        #     continue            
        # counter = 0
        # flag = False
        while True:
            # counter += 1
            node = random.choice(tree.node_list)
            if node.is_leaf or node.parent_node is None:
                continue
            elif len(node.content) < 2 or len(node.content) > 6:
                continue
            # if counter > 100:
            #     flag = True
            #     break
            else:
                break
        # if flag:
        #     continue
        # tokenized_start_idx = node.tokenized_start_position
        # tokenized_end_idx = node.tokenized_end_position
        # for i in range(tokenized_start_idx, tokenized_end_idx):
        #     input_id = tokenizer.encode(tokenized_sentence)
        #     input_id[i:tokenized_end_idx] = tokenizer.mask_token_id
        #     input_ids.append(input_id)
        #     labels.append(torch.where(input_id == tokenizer.mask_token_id, tokenized_sentence, -100))
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
    

device = 'cuda:1'
model = RobertaForMaskedLM.from_pretrained('roberta-large')
model.to(device)

from utils import load

# load tree_list 
train_tree_list = load('/workspace/Hol-CCG/data/tree_list/train_tree_list.pickle')
train_dataset = Tree_Dataset(train_tree_list.tree_list)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
dev_tree_list = load('/workspace/Hol-CCG/data/tree_list/dev_tree_list.pickle')
dev_dataset = Tree_Dataset(dev_tree_list.tree_list)
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# define AdamW optimizer
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5)

# for early stop
max_acc = 0
patience = 3
max_epoch = -1

wandb.init(project='Hol-CCG',name='roberta-large-mlm')
for epoch in range(50):
    with tqdm.tqdm(total=len(train_dataloader)) as pbar:
        pbar.set_description(f'Epoch {epoch}:')
        for input_ids, label, attention_mask in train_dataloader:
            input_ids = input_ids.to(device)
            label = label.to(device)
            attention_mask = attention_mask.to(device)
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
            wandb.log({'train/loss': loss.item(), 'train/acc': acc.item()})

    with tqdm.tqdm(total=len(dev_dataloader)) as pbar:
        pbar.set_description(f'Evaluating:')
        val_loss = 0
        val_acc = 0
        for input_ids, label, attention_mask in dev_dataloader:
            input_ids = input_ids.to(device)
            label = label.to(device)
            attention_mask = attention_mask.to(device)
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
        wandb.log({'val/loss': val_loss, 'val/acc': val_acc})
    
    # check the val_acc for early sotpping of training
    if val_acc > max_acc:
        max_acc = val_acc
        max_epoch = epoch
        patience = 3
        # save model
        torch.save(model.state_dict(), f'/workspace/Hol-CCG/data/model/roberta-mlm.pth')
    else:
        patience -= 1
        if patience == 0:
            print(f'Early stop at epoch {epoch}')
            break




