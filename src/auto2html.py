from tree import TreeList
from tqdm import tqdm
from preprocessing import Converter
import argparse


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_autos', type=str, help='path to auto file')
    args = parser.parse_args()
    return args


def flatten(mathml):
    for i in mathml:
        if isinstance(i, list):
            yield from flatten(i)
        else:
            yield i


def main():
    args = arg_parse()

    converter = Converter()
    converter.convert_and_save(args.path_to_autos, args.path_to_autos.replace('.auto', '.converted'))

    tree_list = TreeList(args.path_to_autos.replace('.auto', '.converted'), 'train')

    with tqdm(total=len(tree_list.tree_list), unit="tree") as pbar:
        pbar.set_description("Setting node information...")
        for tree in tree_list.tree_list:
            tree.root_node = tree.node_list[-1]
            tree.root_node.parent_node = None
            for node in reversed(tree.node_list):
                if not node.is_leaf:
                    if node.num_child == 1:
                        node.child = tree.node_list[node.child_node_id]
                    elif node.num_child == 2:
                        node.left_child = tree.node_list[node.left_child_node_id]
                        node.right_child = tree.node_list[node.right_child_node_id]
            pbar.update(1)

    mathml_list = []
    num_sentence = 0
    with tqdm(total=len(tree_list.tree_list), unit="tree") as pbar:
        pbar.set_description("Converting to html format...")
        for tree in tree_list.tree_list:
            num_sentence += 1
            tree.root_node.mathml = []
            waiting_nodes = [tree.root_node]
            while len(waiting_nodes) > 0:
                node = waiting_nodes.pop(0)
                if node.is_leaf:
                    node.mathml += ['<mfrac><mtext>',
                                    node.content[0],
                                    '</mtext><mtext>',
                                    node.category,
                                    '</mtext></mfrac>']
                else:
                    if node.num_child == 1:
                        node.child.mathml = []
                        node.mathml += ['<mfrac>',
                                        node.child.mathml,
                                        '<mtext>',
                                        node.category,
                                        '</mtext></mfrac>']
                        if node.child.is_leaf:
                            node.child.mathml += ['<mfrac><mtext>',
                                                  node.child.content[0],
                                                  '</mtext><mtext>',
                                                  node.child.category,
                                                  '</mtext></mfrac>']
                        else:
                            waiting_nodes.append(node.child)
                    elif node.num_child == 2:
                        node.left_child.mathml = []
                        node.right_child.mathml = []
                        node.mathml += ['<mfrac><mrow>',
                                        node.left_child.mathml,
                                        node.right_child.mathml,
                                        '</mrow><mtext>',
                                        node.category,
                                        '</mtext></mfrac>']
                        if node.left_child.is_leaf:
                            node.left_child.mathml += ['<mfrac><mtext>',
                                                       node.left_child.content[0],
                                                       '</mtext><mtext>',
                                                       node.left_child.category,
                                                       '</mtext></mfrac>']
                        else:
                            waiting_nodes.append(node.left_child)
                        if node.right_child.is_leaf:
                            node.right_child.mathml += ['<mfrac><mtext>',
                                                        node.right_child.content[0],
                                                        '</mtext><mtext>',
                                                        node.right_child.category,
                                                        '</mtext></mfrac>']
                        else:
                            waiting_nodes.append(node.right_child)
            mathml_list.append('<p>Sentence ID={}</p><math>{}</math>'.format(tree.self_id,
                                                                             ''.join(flatten(tree.root_node.mathml))))
            pbar.update(1)

    html = '''\
    <!doctype html>
    <html lang='en'>
    <head>
    <meta charset='UTF-8'>
    <style>
        body {{
        font-size: 1em;
        }}
    </style>
    <script type="text/javascript"
        src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
    </script>
    </head>
    <body>
    {}
    </body>
    </html>
    '''.format(''.join(mathml_list))
    print(html)


if __name__ == '__main__':
    main()
