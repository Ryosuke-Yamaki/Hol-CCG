from models import Tree_List


class CCG_Category:
    def __init__(self, category):
        # for the complex categories including brackets
        if '(' in category:
            self.category = category[1:-1]
        # for the primitive categories like NP or S
        else:
            self.category = category
        # self.category = category
        self.set_composition_info(self.category)

    def set_composition_info(self, category):
        level = 0
        for idx in range(len(category)):
            char = category[idx]
            if char == '(':
                level += 1
            elif char == ')':
                level -= 1
            if level == 0:
                if char == '\\':
                    self.direction_of_slash = 'L'
                    self.parent_category = category[:idx]
                    self.sibling_category = category[idx + 1:]
                    return
                elif char == '/':
                    self.direction_of_slash = 'R'
                    self.parent_category = category[:idx]
                    self.sibling_category = category[idx + 1:]
                    return
        self.direction_of_slash = None
        self.parent_category = None
        self.sibling_category = None


def compose_categories(left_category, right_category):
    if left_category.direction_of_slash == 'R' and left_category.sibling_category == right_category.category:
        return left_category.parent_category
    elif left_category.direction_of_slash == 'R' and left_category.sibling_category == '(' + right_category.category + ')':
        return left_category.parent_category
    elif right_category.direction_of_slash == 'L' and right_category.sibling_category == left_category.category:
        return right_category.parent_category
    elif right_category.direction_of_slash == 'L' and right_category.sibling_category == '(' + left_category.category + ')':
        return right_category.parent_category


PATH_TO_DIR = "/home/yryosuke0519/"
PATH_TO_TRAIN_DATA = PATH_TO_DIR + "Hol-CCG/data/train.txt"
train_tree_list = Tree_List(PATH_TO_TRAIN_DATA, True)

for tree in train_tree_list.tree_list:
    for node in tree.node_list:
        if not node.is_leaf:
            node.category = None
            print(node.category)


tree_idx = 0
for tree in train_tree_list.tree_list:
    print('*' * 50)
    print('tree_idx = {}'.format(tree_idx))
    node_pair_list = tree.make_node_pair_list()
    while True:
        i = 0
        for node_pair in node_pair_list:
            left_node = node_pair[0]
            right_node = node_pair[1]
            parent_node = tree.node_list[left_node.parent_id]
            if left_node.ready and right_node.ready:
                left_category = CCG_Category(left_node.category)
                right_category = CCG_Category(right_node.category)
                composed_category = compose_categories(left_category, right_category)
                parent_node.category = composed_category
                parent_node.content = left_node.content + ' ' + right_node.content
                parent_node.ready = True
                print('{} + {} ---> {}'.format(left_node.content,
                                               right_node.content, parent_node.content))
                print('{} {} ---> {}'.format(left_category.category,
                                             right_category.category, composed_category))
                print('')
                node_pair_list.remove(node_pair)
        if node_pair_list == []:
            break
    tree_idx += 1
