import ast
import astor
import itertools
import random
import numpy as np
import autopep8
import decorators
import os
from flake8.api import legacy as flake8
from sklearn.preprocessing import MultiLabelBinarizer 
_MLB = MultiLabelBinarizer() 

class convertData:
    def __init__(self, code):
        self.code = code
        self.hash = random.randint(10000000, 100000000)
        self.checker = flake8.get_style_guide(ignore=['E501', 'E741', 'E722'], select=['E']) 
        self.tree, self.program, self.codeprogram, self.encoder, self.elements = self.encode_state(self.code)

        self.Xtrain_, self.ytrain_ = self.generate_actions()

    def encode_state(self, code):
        tree = ast.parse(code)
        # codeprogram = [node for node in ast.walk(tree) if isinstance(node, ast.stmt)]
        codeprogram = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        program = [self.hashcode() for e in codeprogram]
        encoder = _MLB.fit([program])
        return tree, program, codeprogram, encoder, len(program)
    
    def hashcode(self):
        self.hash = self.hash + 1
        return self.hash
    
    def generate_actions(self):
        # Generate all possible subsets of the program
        actions = []
        for r in range(1, len(self.program) + 1):
            actions.extend(itertools.combinations(self.program, r))

        Xtrain = []
        ytrain = []        
        for action in actions:
            _Xtrain = action
            _ytrain = 2
            Xtemp = np.array(self.encoder.transform([_Xtrain])).tolist()
            if (not Xtemp in Xtrain) and np.sum(Xtemp) > 1:
                Xtrain.append(Xtemp)
                ytrain.append(_ytrain)
                
        return np.array(Xtrain), ytrain
    
    def listsort(self, data):
        data_list = list(data)
        all_list = itertools.permutations(data_list)
        return [p for p in all_list]
    
    def convertto(self, state):
        newcode = ""
        for i in range(len(state)):
            if state[i] == 1:
                newcode = newcode + astor.to_source(self.codeprogram[i])
        newcode = autopep8.fix_code(newcode)
        return newcode
    
    def test_syntax(self, state):
        newcode = self.convertto(state)
        file = "___temp_1_2_.py"
        if os.path.isfile(file):
            os.remove(file)            
        f = open(file, 'w')            
        f.write(newcode)   
            
        f.close()
        ttchon = self.test_syntax_file(file)
        return ttchon 

    @decorators.silent
    def test_syntax_file(self, file):           
        report = self.checker.check_files([file])        
        res = report.get_statistics('E') == []
        return res
            
# Example usage:
