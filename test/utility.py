import ast
import astor 
import random

#this method iterates through AST node and collect child node
def setNode( node):
    child = []
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for i in value:
                if isinstance(i, ast.AST):
                    if not hasattr(i, 'is_docstring'):
                        child.append(i)
    return child

class UTILITY():
    def __init__(self):
        self.hashRandom = random.randint(1000000000, 10000000000)
        self.program = []
        self.current_depth = 0


    def hashRan(self):
        self.hashRandom = self.hashRandom + 1
        return self.hashRandom

    
    #this method compute hash of each node and set depth
    def hashTree(self, tree, depth = 0):
            tree.depth = depth
            tree.hash = self.hashRan()
            self.checkDocstring(tree)
            for node in setNode(tree):
                self.hashTree(node, depth + 1)
            return tree

    #this method checks node type
    def checkDocstring(self, node):
            name_class = node.__class__.__name__
            if name_class in ['Module', 'ClassDef', 'FunctionDef']:
                if hasattr(node, 'body'):
                    if isinstance(node.body, list):
                        if len(node.body) > 0:
                            candidate = node.body[0]
                            if candidate.__class__.__name__ == 'Expr':
                                if hasattr(candidate.value, 'value'):
                                    if isinstance(candidate.value.value, str):
                                        candidate.is_docstring = True

    #this  method flatten tree structure using depth first structure
    def flatTree(self, tree, subTree = [], nodeAll = None):
            if tree.hash in subTree or not subTree:
                if nodeAll is None:
                    nodeAll = [tree]
                else:
                    nodeAll.append(tree)
            parent = setNode(tree)
            for child in parent:
                nodeAll = self.flatTree(tree = child, subTree = subTree, nodeAll=nodeAll)
            return nodeAll

    #this method filters out nodes representing docstrings
    def setStatements(self, tree, subTree = None):
            nodeAll = self.flatTree(tree)
            hashAll = [node.hash for node in nodeAll if not hasattr(node, 'is_docstring')]
            if subTree:
                if subTree.hash in hashAll:
                    return self.setStatements(tree = subTree, subTree = None)
                else:
                    return None
            return hashAll    

    def clean(self, ast_tree, target_program, subprogram, depth):
            self.target_program = target_program
            self.subprogram = subprogram
            self.current_depth = depth
            new_tree = self.visit(ast_tree)
            ast.fix_missing_locations(new_tree)
            return new_tree

    def visit(self, node):
            if hasattr(node, 'is_docstring'):
                return node
            if node.hash not in self.target_program\
                and node.hash in self.subprogram\
                and node.depth <= self.current_depth:
                return None
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    replacement = []
                    for item in value:
                        if isinstance(item, ast.AST):
                            new_item = self.visit(item)
                            if new_item is not None:
                                replacement.append(new_item)
                    value[:] = replacement

            return node


        # tree = ast.parse(temp)
        # tree = hashTree(tree)
        # program = setStatements(tree)
        # target_program = program
        # subprogram = program
        # current_depth = 1
        # depth = 1
        # print(program)
        # newTree = clean(ast_tree = tree, target_program = program, subprogram = subprogram, depth = depth)

    # test = setNode(tree)
    # test2 = setNode(newTree)
    # print(test)
    # for i in test:
    #     print(astor.to_source(i))
    # print(test2)
    # for i in test2:
    #     print(astor.to_source(i))`