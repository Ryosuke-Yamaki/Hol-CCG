import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.optim as optim
from utils import evaluate_batch_list, load, Condition_Setter, set_random_seed
from models import Tree_Net
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel, XLNetTokenizer, XLNetModel
from tqdm import tqdm

condition = Condition_Setter(set_embedding_type=False)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:5')
else:
    DEVICE = torch.device('cpu')

# args = sys.argv
args = ['', 'roberta-large', 'True', 'True']
PRETRAINED_MODEL = args[1]
if args[2] == 'False':
    USE_PHRASE_LOSS = False
else:
    USE_PHRASE_LOSS = True
if args[3] == 'False':
    USE_SPAN_LOSS = False
else:
    USE_SPAN_LOSS = True

print("loading transformer model...")
if 'roberta' in PRETRAINED_MODEL:
    TOKENIZER = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL)
    MODEL = RobertaModel.from_pretrained(PRETRAINED_MODEL)
elif 'bert' in PRETRAINED_MODEL:
    TOKENIZER = BertTokenizer.from_pretrained(PRETRAINED_MODEL)
    MODEL = BertModel.from_pretrained(PRETRAINED_MODEL)
elif 'xlnet' in PRETRAINED_MODEL:
    TOKENIZER = XLNetTokenizer.from_pretrained(PRETRAINED_MODEL)
    MODEL = XLNetModel.from_pretrained(PRETRAINED_MODEL)
if 'base' in PRETRAINED_MODEL:
    EMBEDDING_DIM = 768
elif 'large' in PRETRAINED_MODEL:
    EMBEDDING_DIM = 1024

train_tree_list = load(condition.path_to_train_tree_list)
dev_tree_list = load(condition.path_to_dev_tree_list)
train_tree_list.device = DEVICE
dev_tree_list.device = DEVICE

train_tree_list.set_info_for_training(TOKENIZER)
dev_tree_list.set_info_for_training(TOKENIZER)

NUM_WORD_CAT = len(train_tree_list.word_category_vocab.stoi)
NUM_PHRASE_CAT = len(train_tree_list.phrase_category_vocab.stoi)
NUM_POS_TAG = len(train_tree_list.pos_tag_vocab.stoi)

# hyper parameters
EPOCHS = 10
TRAIN_EMBEDDER = True

if USE_PHRASE_LOSS:
    phrase_loss_weight = 1.0
else:
    phrase_loss_weight = 0.0
if USE_SPAN_LOSS:
    span_loss_weight = 1.0
else:
    span_loss_weight = 0.0


def objective(trial):
    SEED = trial.suggest_int("SEED", 0, 1e+5)
    BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [16, 32])
    BASE_LR = trial.suggest_categorical("BASE_LR", [2e-4, 1e-4, 5e-5])
    FT_LR = trial.suggest_categorical("FT_LR", [2e-5, 1e-5, 5e-6])
    DROPOUT = trial.suggest_categorical("DROPOUT", [0.1, 0.2, 0.4])
    MODEL_DIM = trial.suggest_categorical("MODEL_DIM", [256, 512, 1024])
    set_random_seed(SEED)
    tree_net = Tree_Net(
        num_word_cat=NUM_WORD_CAT,
        num_phrase_cat=NUM_PHRASE_CAT,
        num_pos_tag=NUM_POS_TAG,
        model=MODEL,
        tokenizer=TOKENIZER,
        train_encoder=TRAIN_EMBEDDER,
        embedding_dim=EMBEDDING_DIM,
        model_dim=MODEL_DIM,
        ff_dropout=DROPOUT,
        device=DEVICE).to(DEVICE)
    cat_criteria = nn.CrossEntropyLoss()
    span_criteria = nn.BCELoss()
    optimizer = optim.AdamW([{'params': tree_net.base_params},
                            {'params': tree_net.model.parameters(), 'lr': FT_LR}], lr=BASE_LR)
    dev_batch_list = dev_tree_list.make_batch(BATCH_SIZE)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    for epoch in range(1, EPOCHS + 1):
        tree_net.train()
        train_batch_list = train_tree_list.make_batch(BATCH_SIZE)
        epoch_word_loss = 0.0
        epoch_phrase_loss = 0.0
        epoch_span_loss = 0.0
        num_batch = 0
        with tqdm(total=len(train_batch_list), unit="batch") as pbar:
            pbar.set_description(f"Epoch[{epoch}/{EPOCHS}]")
            for batch in train_batch_list:
                optimizer.zero_grad()
                word_output, phrase_output, span_output, word_label, phrase_label, span_label = tree_net(
                    batch)
                span_output = torch.sigmoid(span_output)
                word_loss = cat_criteria(word_output, word_label)
                phrase_loss = cat_criteria(phrase_output, phrase_label)
                span_loss = span_criteria(span_output, span_label)

                loss_to_backward = word_loss + phrase_loss_weight * phrase_loss + span_loss_weight * span_loss
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

        tree_net.eval()
        dev_stat = evaluate_batch_list(dev_batch_list, tree_net)

        trial.report(dev_stat['phrase_acc'], epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return dev_stat['phrase_acc']


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=1))
    study.optimize(objective, n_trials=300, gc_after_trial=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
