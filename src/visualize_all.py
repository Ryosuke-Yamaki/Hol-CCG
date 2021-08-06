from utils import load, dump, set_random_seed, Condition_Setter
from sklearn.decomposition import PCA
from collections import Counter
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_n_np_s_category_id(category_vocab):
    N = []
    NP = []
    S = []
    for k, v in category_vocab.stoi.items():
        if 'NP' in k and ('/' not in k and '\\' not in k):
            NP.append(v)
        elif 'N' in k and ('/' not in k and '\\' not in k):
            N.append(v)
        elif 'S' in k and ('/' not in k and '\\' not in k):
            S.append(v)
    return N, NP, S


def check_include(counter, category_list):
    for category_id in category_list:
        if category_id in counter:
            return True
    return False


def prepare_vector_list(tree_list):
    vector_list = []
    content_to_idx = {}
    idx_dict = {}
    idx = 0
    for tree in tree_list.tree_list:
        for node in tree.node_list:
            if tuple(node.content_id) not in content_to_idx:
                content_to_idx[tuple(node.content_id)] = idx
                vector_list.append(node.vector.detach().numpy()[0])
                category_counter = Counter()
                category_counter[node.category_id] += 1
                idx_dict[idx] = {'content': node.content_id,
                                 'category_counter': category_counter}
                idx += 1
            else:
                idx_dict[content_to_idx[tuple(node.content_id)]
                         ]['category_counter'][node.category_id] += 1
    N, NP, S = get_n_np_s_category_id(tree_list.category_vocab)
    vis_dict = {'N': [], 'NP': [], 'S': [], 'Word': [], 'Phrase': []}
    color_list = []

    for k, v in idx_dict.items():
        idx = k
        category_counter = v['category_counter']
        if check_include(category_counter, N):
            vis_dict['N'].append(idx)
            color_list.append('tab:blue')
        elif check_include(category_counter, NP):
            vis_dict['NP'].append(idx)
            color_list.append('tab:orange')
        elif check_include(category_counter, S):
            vis_dict['S'].append(idx)
            color_list.append('tab:green')
        elif len(v['content']) == 1:
            vis_dict['Word'].append(idx)
            color_list.append('tab:red')
        else:
            vis_dict['Phrase'].append(idx)
            color_list.append('tab:purple')
    return vector_list, idx_dict, vis_dict, color_list


condition = Condition_Setter(set_embedding_type=False)
calculate = int(input('new calculation(0) or load(1): '))
if calculate not in [0, 1]:
    print('Error: new calculation or load')
    exit()

device = torch.device('cpu')

set_random_seed(0)

print('loading tree list...')
test_tree_list = load(condition.path_to_test_tree_list)

for embedding_type in ['GloVe', 'random']:
    if embedding_type == 'GloVe':
        dim_list = [50, 100, 300]
    else:
        dim_list = [10, 50, 100, 300]
    for embedding_dim in dim_list:
        print('-----{}_{}d-----'.format(embedding_type, embedding_dim))
        condition.embedding_type = embedding_type
        condition.embedding_dim = embedding_dim
        condition.set_path()
        if calculate == 0:
            if condition.embedding_type == "random":
                tree_net = torch.load(condition.path_to_model,
                                      map_location=torch.device('cpu'))
            else:
                tree_net = torch.load(
                    condition.path_to_model_with_regression,
                    map_location=torch.device('cpu'))
            tree_net.eval()
            embedding = tree_net.embedding
            test_tree_list.set_vector(embedding)
            vector_list, idx_dict, vis_dict, color_list = prepare_vector_list(
                test_tree_list)
            dump(vis_dict, condition.path_to_vis_dict)
            dump(idx_dict, condition.path_to_idx_dict)
            dump(color_list, condition.path_to_color_list)
        for method_id in [0, 1]:
            for visualize_dim in [2, 3]:
                if calculate == 0:
                    if method_id == 0:
                        method = TSNE(n_components=visualize_dim)
                        path_to_visualize_weight = condition.path_to_visualize_weight + \
                            "_{}d_t-SNE.pickle".format(visualize_dim)
                        path_to_map = condition.path_to_map + \
                            "_{}d_t-SNE.pdf".format(visualize_dim)
                        print("t-SNE_{}d working.....".format(visualize_dim))
                    elif method_id == 1:
                        method = PCA(n_components=visualize_dim)
                        path_to_visualize_weight = condition.path_to_visualize_weight + \
                            "_{}d_PCA.pickle".format(visualize_dim)
                        path_to_map = condition.path_to_map + \
                            "_{}d_PCA.pdf".format(visualize_dim)
                        print("PCA_{}d working.....".format(visualize_dim))

                    embedded = method.fit_transform(vector_list)
                    dump(embedded, path_to_visualize_weight)
                    if method == 1:
                        print("experined variance ratio = ",
                              method.explained_variance_ratio_)

                else:
                    if method_id == 0:
                        path_to_visualize_weight = condition.path_to_visualize_weight + \
                            "_{}d_t-SNE.pickle".format(visualize_dim)
                        path_to_map = condition.path_to_map + "_{}d_t-SNE.pdf".format(visualize_dim)
                    else:
                        path_to_visualize_weight = condition.path_to_visualize_weight + \
                            "_{}d_PCA.pickle".format(visualize_dim)
                        path_to_map = condition.path_to_map + "_{}d_PCA.pdf".format(visualize_dim)
                    embedded = load(path_to_visualize_weight)
                    vis_dict = load(condition.path_to_vis_dict)
                    idx_dict = load(condition.path_to_idx_dict)
                    color_list = load(condition.path_to_color_list)

                fig0 = plt.figure(figsize=(10, 10))
                if visualize_dim == 2:
                    ax = fig0.add_subplot()
                    for k, v in vis_dict.items():
                        ax.scatter(embedded[v][:, 0],
                                   embedded[v][:, 1], s=1, label=k)
                elif visualize_dim == 3:
                    ax = fig0.add_subplot(projection='3d')
                    for k, v in vis_dict.items():
                        ax.scatter(embedded[v][:, 0], embedded[v][:, 1],
                                   embedded[v][:, 2], s=1, label=k)
                ax.legend(fontsize='large')
                fig0.savefig(path_to_map)
                plt.close(fig0)
