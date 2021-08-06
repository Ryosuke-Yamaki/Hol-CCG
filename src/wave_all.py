from utils import load, single_circular_correlation, set_random_seed, Condition_Setter
import torch
import matplotlib.pyplot as plt

condition = Condition_Setter(set_embedding_type=False)
sentence = input('input 4 words sentence: ').lower()
sentence = sentence.split()
w0 = sentence[0]
w1 = sentence[1]
w2 = sentence[2]
w3 = sentence[3]

device = torch.device('cpu')
set_random_seed(0)

print('loading tree list...')
test_tree_list = load(condition.path_to_test_tree_list)
vocab = test_tree_list.content_vocab

for embedding_type in ['GloVe', 'random']:
    if embedding_type == 'GloVe':
        dim_list = [50, 100, 300]
    else:
        dim_list = [10, 50, 100, 300]
    for embedding_dim in dim_list:
        condition.embedding_type = embedding_type
        condition.embedding_dim = embedding_dim
        condition.set_path()
        if condition.embedding_type == 'random':
            tree_net = torch.load(condition.path_to_model, map_location=device)
        else:
            tree_net = torch.load(condition.path_to_model_with_regression, map_location=device)
        tree_net.eval()
        embedding = tree_net.embedding

        wave_list = {}
        wave_list[w0] = embedding(torch.tensor(vocab[w0]))
        wave_list[w1] = embedding(torch.tensor(vocab[w1]))
        wave_list[w2] = embedding(torch.tensor(vocab[w2]))
        wave_list[w3] = embedding(torch.tensor(vocab[w3]))
        wave_list['{}_{}'.format(w0, w1)] = single_circular_correlation(
            wave_list[w0], wave_list[w1])
        wave_list['{}_{}'.format(w2, w3)] = single_circular_correlation(
            wave_list[w2], wave_list[w3])
        wave_list['{}_{}_{}_{}'.format(w0, w1, w2, w3)] = single_circular_correlation(
            wave_list['{}_{}'.format(w0, w1)], wave_list['{}_{}'.format(w2, w3)])

        x = range(0, condition.embedding_dim)
        for k, v in wave_list.items():
            fig = plt.figure()
            plt.plot(x, v.detach().numpy())
            plt.title(condition.embedding_type + '_' +
                      str(condition.embedding_dim) + 'd - ' + k.replace('_', ' '))
            fig.savefig(condition.path_to_wave + '_' + k)
            plt.close(fig)
