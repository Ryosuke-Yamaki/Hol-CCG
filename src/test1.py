import time
from models import Tree_List

path_to_train_data = '/home/yryosuke0519/CCGbank/converted/train.txt'
path_to_dev_data = '/home/yryosuke0519/CCGbank/converted/dev.txt'
path_to_test_data = '/home/yryosuke0519/CCGbank/converted/test.txt'
start = time.time()
train_tree_list = Tree_List(path_to_train_data)
# dev_tree_list = Tree_List(path_to_dev_data)
# test_tree_list = Tree_List(path_to_test_data)
end = time.time()
print(end - start)

# for tree_list in [train_tree_list, dev_tree_list, test_tree_list]:
for tree in train_tree_list.tree_list:
    if len(tree.node_list) == 1:
        for composition_info in tree.composition_info:
            if len(composition_info) == 2:
                child_node = tree.node_list[composition_info[0]]
                parent_node = tree.node_list[composition_info[1]]
                parent_node.content = child_node.content
                print('{}:{} ---> {}:{}'.format(child_node.content,
                                                child_node.category, parent_node.content, parent_node.category))
            else:
                left_child_node = tree.node_list[composition_info[0]]
                right_child_node = tree.node_list[composition_info[1]]
                parent_node = tree.node_list[composition_info[2]]
                parent_node.content = left_child_node.content + ' ' + right_child_node.content
                print('{}:{}, {}:{} ---> {}:{}'.format(left_child_node.content,
                                                       left_child_node.category,
                                                       right_child_node.content,
                                                       right_child_node.category,
                                                       parent_node.content,
                                                       parent_node.category))

    # print('')
print('total={}'.format(len(train_tree_list.content_to_id)))
num = 0
for word in dev_tree_list.content_to_id.keys():
    if word not in train_tree_list.content_to_id:
        num += 1
print('total={}'.format(len(dev_tree_list.content_to_id)))
print('not_exist={}'.format(num))
print('ratio={}'.format(num / len(dev_tree_list.content_to_id)))

num = 0
for word in test_tree_list.content_to_id.keys():
    if word not in train_tree_list.content_to_id:
        num += 1
print('total={}'.format(len(test_tree_list.content_to_id)))
print('not_exist={}'.format(num))
print('ratio={}'.format(num / len(test_tree_list.content_to_id)))
