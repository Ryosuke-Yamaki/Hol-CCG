from sys import path
from utils import load


class Combinator:
    def __init__(self, path_to_grammar, min_freq=10):
        self.path_to_grammar = path_to_grammar
        self.punc_cats = {
            'left_,': [],
            'left_:;': [],
            'left_LRB': [],
            'right_,': [],
            'right_:;': [],
            'right_.': [],
            'right_RRB': []}
        self.coord_cats = {'left_,': [], 'left_;': []}
        self.u_type_change = {}
        with open(path_to_grammar + 'punctuation.grammar', 'r') as f:
            list = f.readlines()
        for line in list:
            line = line.split()
            if line[2] == ',':
                self.punc_cats['left_,'].append(line[3])
            elif line[2] == ':' or line[2] == ';':
                self.punc_cats['left_:;'].append(line[3])
            elif line[2] == 'LRB':
                self.punc_cats['left_LRB'].append(line[3])
            elif line[3] == ',':
                self.punc_cats['right_,'].append(line[2])
            elif line[3] == ':' or line[3] == ';':
                self.punc_cats['right_:;'].append(line[2])
            elif line[3] == '.':
                self.punc_cats['right_.'].append(line[2])
            elif line[3] == 'RRB':
                self.punc_cats['right_RRB'].append(line[2])
        with open(path_to_grammar + 'coordination.grammar', 'r') as f:
            list = f.readlines()
        for line in list:
            line = line.split()
            if line[2] == ',':
                self.coord_cats['left_,'].append(line[3])
            elif line[2] == ';':
                self.coord_cats['left_;'].append(line[3])
        with open(path_to_grammar + 'unary_type_change.grammar', 'r') as f:
            list = f.readlines()
        for line in list:
            line = line.split()
            parent = line[0]
            child = line[2]
            if child in self.u_type_change:
                self.u_type_change[child].append(parent)
            else:
                self.u_type_change[child] = [parent]
        self.head_info = load(path_to_grammar + 'head_info.grammar')
        self.extract_rule(min_freq)

    def extract_rule(self, min_freq):
        self.binary_rule = {}
        self.unary_rule = {}
        with open(self.path_to_grammar + "CCGbank.02-21.grammar", 'r') as f:
            rules = f.readlines()
        for rule in rules:
            rule = rule.split()
            # for binary rule
            if len(rule) == 6:
                freq = int(rule[0])
                parent = rule[2]
                left = rule[4]
                right = rule[5]
                composed_parents, types = self.apply_binary(left, right)
                # for the basic combinatory rules
                if parent in composed_parents:
                    type = types[composed_parents.index(parent)]
                    if (left, right) in self.binary_rule:
                        self.binary_rule[(left, right)].append([parent, type])
                    else:
                        self.binary_rule[(left, right)] = [[parent, type]]
                elif freq >= min_freq:
                    if (left, right) in self.binary_rule:
                        self.binary_rule[(left, right)].append([parent, 'other'])
                    else:
                        self.binary_rule[(left, right)] = [[parent, 'other']]
            # for unary rule
            elif len(rule) == 5:
                freq = int(rule[0])
                parent = rule[2]
                child = rule[4]
                composed_parent, types = self.apply_unary(child)
                if parent in composed_parent:
                    type = types[composed_parent.index(parent)]
                    if child in self.unary_rule:
                        self.unary_rule[child].append([parent, type])
                    else:
                        self.unary_rule[child] = [[parent, type]]
                elif freq >= min_freq:
                    if child in self.unary_rule:
                        self.unary_rule[child].append([parent, 'other'])
                    else:
                        self.unary_rule[child] = [[parent, 'other']]

    def apply_binary(self, left, right):
        left = self.get_cat_info(left)
        right = self.get_cat_info(right)
        # apply forward application
        if left['lr'] == 'right' and left['arg'] == right['cat']:
            parent = left['target']
            parent = [parent]
            type = ['fa']
        # apply backward application
        elif right['lr'] == 'left' and left['cat'] == right['arg']:
            parent = right['target']
            parent = [parent]
            type = ['ba']
        # apply forward compostion
        elif left['lr'] == 'right' and right['lr'] == 'right' and left['arg'] == right['target']:
            if '/' in left['target'] or '\\' in left['target']:
                parent = '(' + left['target'] + ')'
            else:
                parent = left['target']
            parent += '/'
            if '/' in right['arg'] or '\\' in right['arg']:
                parent += '(' + right['arg'] + ')'
            else:
                parent += right['arg']
            parent = [parent]
            type = ['fc']
        # apply backward composition
        elif left['lr'] == 'left' and right['lr'] == 'left' and left['target'] == right['arg']:
            if '/' in right['target'] or '\\' in right['target']:
                parent = '(' + right['target'] + ')'
            else:
                parent = right['target']
            parent += '\\'
            if '/' in left['arg'] or '\\' in left['arg']:
                parent += '(' + left['arg'] + ')'
            else:
                parent += left['arg']
            parent = [parent]
            type = ['bc']
        # apply backward cross composition
        elif left['lr'] == 'right' and right['lr'] == 'left' and left['target'] == right['arg']:
            if '/' in right['target'] or '\\' in right['target']:
                parent = '(' + right['target'] + ')'
            else:
                parent = right['target']
            parent += '/'
            if '/' in left['arg'] or '\\' in left['arg']:
                parent += '(' + left['arg'] + ')'
            else:
                parent += left['arg']
            parent = [parent]
            type = ['bxc']

        # apply conjugation
        elif left['cat'] == 'conj':
            if '[conj]' in right['cat']:
                parent = right['cat']
            else:
                parent = right['cat'] + '[conj]'
            parent = [parent]
            type = ['conj']
            if right['cat'] == 'N':
                parent.append(right['cat'])
                type.append('conj')

        elif left['cat'] in [',', ':', ';', 'LRB']:
            # apply punctuation rules
            if ((left['cat'] == ',' and right['cat'] in self.punc_cats['left_,'])
                or (left['cat'] in [':', ';'] and right['cat'] in self.punc_cats['left_:;'])
                    or (left['cat'] == 'LRB' and right['cat'] in self.punc_cats['left_LRB'])):
                parent = [right['cat']]
                type = ['punc']
                if left['cat'] == ',':
                    # apply binary type changing
                    if right['cat'] == 'NP':
                        parent.append('(S\\NP)\\(S\\NP)')
                        type.append('btc')
                    # apply coordination
                    if right['cat'] in self.coord_cats['left_,']:
                        parent.append(right['cat'] + '\\' + right['cat'])
                        type.append('coord')
                # apply coordination
                elif left['cat'] == ';':
                    if right['cat'] in self.coord_cats['left_;']:
                        parent.append(right['cat'] + '\\' + right['cat'])
                        type.append('coord')
            else:
                parent = []
                type = []

        elif right['cat'] in [',', ':', ';', '.', 'RRB']:
            # apply punctuation
            if ((right['cat'] == ',' and left['cat'] in self.punc_cats['right_,'])
                or (right['cat'] in [':', ';'] and left['cat'] in self.punc_cats['right_:;'])
                or (right['cat'] in ['.'] and left['cat'] in self.punc_cats['right_.'])
                    or (right['cat'] in ['RRB'] and left['cat'] in self.punc_cats['right_RRB'])):
                parent = [left['cat']]
                type = ['punc']
                if right['cat'] == ',':
                    # apply binary type changing
                    if left['cat'] == "NP":
                        parent.append("S/S")
                        type.append('btc')
                    elif left['cat'] == "S[dcl]/S[dcl]":
                        parent.extend(["S/S", "(S\\NP)\\(S\\NP)", "(S\\NP)/(S\\NP)"])
                        type.extend(['btc', 'btc', 'btc'])
                    elif left['cat'] == "S[dcl]/S[dcl]":
                        parent.extend(["S\\S", "S/S"])
                        type.extend(['btc', 'btc'])
            else:
                parent = []
                type = []

        elif left['cat'] == 'NP' and right['cat'] == 'NP[conj]':
            parent = ['NP']
            type = ['other']
        elif left['cat'] == 'S[dcl]' and right['cat'] == 'S[dcl][conj]':
            parent = ['S[dcl]']
            type = ['other']
        else:
            parent = []
            type = []
        return parent, type

    def apply_unary(self, child):
        child = self.get_cat_info(child)
        parent = []
        type = []
        # apply type-raising
        if child['cat'] == 'NP':
            parent.extend(["S/(S\\NP)",
                           "(S\\NP)\\((S\\NP)/NP)",
                           "((S\\NP)/NP)\\(((S\\NP)/NP)/NP)",
                           "((S\\NP)/(S[to]\\NP))\\(((S\\NP)/(S[to]\\NP))/NP)",
                           "((S\\NP)/PP)\\(((S\\NP)/PP)/NP)",
                           "((S\\NP)/(S[adj]\\NP))\\(((S\\NP)/(S[adj]\\NP))/NP)"])
            type.extend(["tr", "tr", "tr", "tr", "tr", "tr"])
        elif child['cat'] == 'PP':
            parent.append("(S\\NP)\\((S\\NP)/PP)")
            type.append("tr")
        elif child['cat'] == 'S[adj]\\NP':
            parent.append("(S\\NP)\\((S\\NP)/(S[adj]\\NP))")
            type.append('tr')
        # apply type-changing
        if child['cat'] in self.u_type_change:
            parent.extend(self.u_type_change[child['cat']])
            type.extend(["utc" for i in range(len(self.u_type_change[child['cat']]))])
        return parent, type

    def get_cat_info(self, cat):
        level = 0
        is_primitive = True
        lr = None
        arg = None
        target = None
        for i in range(len(cat)):
            c = cat[i]
            if c == '(':
                level += 1
            elif c == ')':
                level -= 1
            elif (c == '\\' or c == '/') and level == 0:
                is_primitive = False
                arg = cat[i + 1:]
                target = cat[:i]
                if arg[0] == '(':
                    arg = arg[1:-1]
                if target[0] == '(':
                    target = target[1:-1]
                if c == '\\':
                    lr = 'left'
                elif c == '/':
                    lr = 'right'
        info = {'cat': cat, 'is_primitive': is_primitive, 'arg': arg, 'target': target, 'lr': lr}
        return info
