import torch
from models import Tree_List, Tree_Net, Condition_Setter


class Analyzer:
    def __init__(self, tree_list, tree_net):
        self.tree_list = tree_list
        self.id_to_category = tree_list.id_to_category
        self.tree_net = tree_net

    def analyze(self, THRESHOLD, target_tree_id=None):
        if target_tree_id is None:
            tree_list = self.tree_list.tree_list
        else:
            tree_list = []
            for tree_id in target_tree_id:
                tree_list.append(self.tree_list.tree_list[tree_id])
        for tree in tree_list:
            leaf_node_info = tree.leaf_node_info
            label_list = tree.label_list
            composition_info = tree.composition_info
            output = self.tree_net(leaf_node_info, composition_info)
            acc = self.cal_acc(output, label_list)
            print('{}:{}'.format(tree.self_id, tree.sentence))
            print('acc: {}'.format(acc))
            print('*' * 50)
            for node in tree.node_list:
                content = node.content
                prob = output[node.self_id]
                pred_category_id = []
                pred_category = []
                for i in range(len(prob)):
                    if prob[i] >= THRESHOLD:
                        pred_category_id.append(i)
                        pred_category.append(self.id_to_category[i])
                if pred_category == []:
                    pred_category_id.append(int(torch.argmax(prob)))
                    pred_category.append(self.id_to_category[int(torch.argmax(prob))])

                content = []
                for content_id in node.content_id:
                    content.append(self.tree_list.id_to_content[content_id])
                print('content: {}'.format(' '.join(content)))
                true_category = []
                for category_id in node.possible_category_id:
                    true_category.append(self.id_to_category[category_id])
                print('true category: {}'.format(true_category))
                print('pred category: {}'.format(pred_category))
                if set(node.possible_category_id) == set(pred_category_id):
                    print('True')
                else:
                    print('False')
                print()
            print('*' * 50)

    def cal_acc(self, output, label_list):
        num_correct = 0
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if output[i][j] >= 0.5:
                    output[i][j] = 1.0
                else:
                    output[i][j] = 0.0
            if torch.count_nonzero(output[i] == label_list[i]) == output.shape[1]:
                num_correct += 1
        return num_correct / output.shape[0]


PATH_TO_DIR = "/home/yryosuke0519/Hol-CCG/"

condition = Condition_Setter(PATH_TO_DIR)

# initialize tree_list from toy_data
train_tree_list = Tree_List(condition.path_to_train_data, condition.REGULARIZED)
test_tree_list = Tree_List(condition.path_to_test_data, condition.REGULARIZED)
# match the vocab and category between train and test data
test_tree_list.replace_vocab_category(train_tree_list)

tree_net = Tree_Net(test_tree_list, condition.embedding_dim)
tree_net = torch.load(condition.path_to_model)
tree_net.eval()
trained_weight_matrix = tree_net.embedding.weight

THRESHOLD = 0.3

target_tree_id = input("target tree id(default=all): ")
if target_tree_id != "":
    target_tree_id = [int(x) for x in target_tree_id.split(",")]
else:
    target_tree_id = None

analyzer = Analyzer(test_tree_list, tree_net)
analyzer.analyze(THRESHOLD, target_tree_id)
