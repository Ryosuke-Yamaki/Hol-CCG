import torch
from models import Tree_List, Tree_Net, Condition_Setter
from utils import load_weight_matrix


class Analyzer:
    def __init__(self, tree_list, tree_net):
        self.tree_list = tree_list
        self.id_to_category = tree_list.id_to_category
        self.tree_net = tree_net

    def analyze(self, target_tree_id=None):
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
            print(tree.sentence)
            print('acc: {}'.format(acc))
            print('*' * 50)
            for node in tree.node_list:
                content = node.content
                true_category = node.category
                pred_category = self.id_to_category[int(torch.argmax(output[node.self_id]))]
                print('content: {}'.format(content))
                print('true category: {}'.format(true_category))
                print('pred category: {}'.format(pred_category))
                if true_category == pred_category:
                    print('True')
                else:
                    print('False')
                print()
            print('*' * 50)

    def cal_acc(self, output, label_list):
        pred = torch.argmax(output, dim=1)
        num_correct = torch.count_nonzero(pred == label_list)
        return num_correct / output.shape[0]


PATH_TO_DIR = "/home/yryosuke0519/Hol-CCG/"

condition = Condition_Setter(PATH_TO_DIR)

# initialize tree_list from toy_data
train_tree_list = Tree_List(condition.path_to_train_data, condition.REGULARIZED)
test_tree_list = Tree_List(condition.path_to_test_data, condition.REGULARIZED)
test_tree_list.content_to_id = train_tree_list.content_to_id
test_tree_list.category_to_id = train_tree_list.category_to_id
test_tree_list.set_content_category_id()
test_tree_list.set_info_for_training()

weight_matrix = torch.tensor(
    load_weight_matrix(
        condition.path_to_initial_weight_matrix,
        condition.REGULARIZED))
tree_net = Tree_Net(test_tree_list, weight_matrix)
tree_net.load_state_dict(torch.load(condition.path_to_model))
tree_net.eval()

analyzer = Analyzer(test_tree_list, tree_net)
analyzer.analyze()
