from utils import load, set_random_seed, Condition_Setter
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
                idx_dict[idx] = {'content': node.content_id, 'category_counter': category_counter}
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


def interactive_visualize_test(
        visualize_dim,
        embedded,
        idx_dict,
        color_list,
        content_vocab,
        category_vocab):
    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        content_id = idx_dict[ind["ind"][0]]['content']
        text = []
        for id in content_id:
            text.append(content_vocab.itos[id])
        text = ' '.join(text)
        for id in list(idx_dict[ind["ind"][0]]['category_counter'].keys()):
            text += ('\n' + category_vocab.itos[id])
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('white')

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig = plt.figure(figsize=(10, 10))
    if visualize_dim == 2:
        ax = fig.add_subplot()
        sc = ax.scatter(embedded[:, 0], embedded[:, 1], c=color_list, s=3)
    elif visualize_dim == 3:
        ax = fig.add_subplot(projection='3d')
        sc = ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2], c=color_list, s=3)
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", hover)
    return fig


condition = Condition_Setter()
method = int(input('t-SNE(0) or PCA(1): '))
visualize_dim = int(input('2d(2) or 3d(3): '))

device = torch.device('cpu')

set_random_seed(0)
print('loading tree list...')
test_tree_list = load(condition.path_to_test_tree_list)

if condition.embedding_type == "random":
    tree_net = torch.load(condition.path_to_model, map_location=torch.device('cpu'))
else:
    tree_net = torch.load(condition.path_to_model_with_regression, map_location=torch.device('cpu'))
tree_net.eval()
embedding = tree_net.embedding

test_tree_list.set_vector(embedding)

vector_list, idx_dict, vis_dict, color_list = prepare_vector_list(test_tree_list)

if method == 0:
    method = TSNE(n_components=visualize_dim)
    path_to_map = condition.path_to_map + "_{}d_t-SNE.png".format(visualize_dim)
    print("t-SNE working.....")
else:
    method = PCA(n_components=visualize_dim)
    path_to_map = condition.path_to_map + "_{}d_PCA.png".format(visualize_dim)
    print("PCA working.....")

embedded = method.fit_transform(vector_list)
if method == 1:
    print("experined variance ratio = ", method.explained_variance_ratio_)

fig0 = plt.figure(figsize=(10, 10))
if visualize_dim == 2:
    ax = fig0.add_subplot()
    for k, v in vis_dict.items():
        ax.scatter(embedded[v][:, 0], embedded[v][:, 1], s=1, label=k)
    ax.legend(fontsize='large')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    fig0.savefig(path_to_map)
elif visualize_dim == 3:
    ax = fig0.add_subplot(projection='3d')
    for k, v in vis_dict.items():
        ax.scatter(embedded[v][:, 0], embedded[v][:, 1], embedded[v][:, 2], s=1, label=k)
    ax.legend(fontsize='large')
    fig0.savefig(path_to_map)

fig1 = interactive_visualize_test(
    visualize_dim,
    embedded,
    idx_dict,
    color_list,
    test_tree_list.content_vocab,
    test_tree_list.category_vocab)
plt.show()
