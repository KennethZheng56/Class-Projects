"""
Update this file with your implementations
"""
# in addition to the following modules, you can import any modules from Python's **standard library only**.
# For any non-standard modules that require separate installation, you must obtain instructor approval in advance of the deadline.
import numpy as np
import math
import torch as tr
import transformers as tf
from utils import setup_llm, ProbabilityEstimator # can't use ProbEstimator

### you can write any helper functions you like here

class Node:
    
    def __init__(self, value, prob, child=None, ):
        self.value = value
        self.child = []
        self.visits = 0
        self.score = 0
        self.estimate = 0
        self.words = []
        self.prob = prob
        if child != None:
            self.child = child

    def add_child(self, child):
        self.child = child

    def add_visits(self):
        self.visits += 1

    def add_score(self, score):
        self.score += score
    
    def set_estimate(self):
        if self.visits != 0:
            self.estimate = self.score / self.visits

    def prune(self, option=0):
        if option == 0:
            if len(self.child) > 4:
                b = []
                for j in range(3):
                    mi = None
                    for i in self.child:
                        if mi:
                            if i.estimate > mi.estimate:
                                mi = i
                        else:
                            mi = i
                    b.append(mi)
                    self.child.remove(mi)
                self.child = b
            if len(self.child) > 1:
                mi = None
                for i in self.child:
                    if not mi:
                        mi = i
                    else:
                        if i.estimate > mi.estimate:
                            mi = i
                self.child = [mi]
        elif option == 1:
            if len(self.child) > 1:
                mi = None
                for i in self.child:
                    if not mi:
                        mi = i
                    else:
                        if i.score > mi.score:
                            mi = i
                self.child = [mi]
        elif option == 2:
            if len(self.child) > 1:
                mi = None
                for i in self.child:
                    if not mi:
                        mi = i
                    else:
                        if i.visits > mi.visits:
                            mi = i
                self.child = [mi]
        

    def print(self, prev):
        # root
        prev = (prev + "||") * len(self.child)
        b = prev.split("||")
        ph = []
        a = []
        d = ""
        for c in range(len(self.child)):
            word = b[c] + self.child[c].value
            ph.append(word)
            if len(self.child[c].child) > 0:
                a.append(self.child[c].print(word))
            else:
                d = word
                print(d, self.child[c].score)
                
        return 0

class Test:
    root:Node
    def __init__(self, root):
        self.root = root
        
    def attach_child(self, path:str, child):
        a = self.root
        b = []
        p = path.strip().split(" ")
        if len(a.child) == 0:
            self.root = child
        else:
            for v in p:
                ph = None
                for i in range(len(a.child)):
                    c = a.child[i]
                    if v in c.value:
                        b.append(i)
                        ph = c
                        break
                if ph:
                    a = ph
            a = child
        # self.root.print(self.root.value)


def choose_child(root:Node, children: list[Node], option = 0): # exploration (choose lowest # of visits)
    # guarenteed to have child
    top = None
    if option == 0: # exploration
        for child in children:
            if not top:
                top = child
            else:
                if child.visits < top.visits:
                    top = child 
    elif option == 1: # UCT
        for child in children:
            a = 0
            b = 0
            if not top:
                top = child
            else:
                if top.visits == 0: a = 1
                else:
                    a = top.visits
                if child.visits == 0: b = 1
                else: b = child.visits
                te = top.estimate + 5 * (math.log(root.visits) / a) ** 0.5 #ln
                ce = child.estimate + 5 * (math.log(root.visits) / b) ** 0.5 #ln
                if te < ce:
                    top = child
        top.add_visits()
        top.add_score(top.prob)
        top.set_estimate()
    return top

def MCTS(root:Node):
    if len(root.child) == 0:
        result = root.prob
    else:
        result, r = MCTS(choose_child(root, root.child, 1))
        root.add_visits()
        root.add_score(result)
        root.set_estimate()
    return result, root

def base(prompt, num_tokens, root=None, length=0, parent=None):
    if not root:
        root = Node(prompt, 1)

    done = False
    # use same setup as evaluation
    tokenizer, model, mask = setup_llm()

    # initialize string to prompt before appending tokens
    string = prompt

    # generate child
    token_ids = tokenizer(string, return_tensors="pt").input_ids # (1, seq length)
    # done condition
    if token_ids.shape[1] == num_tokens-1: done = True

    # score the valid next tokens
    outputs = model(input_ids=token_ids)
    logits = outputs.logits[0,-1,:] # (vocab size,)
    logits = logits + mask

    # get probability table / policy
    probs = tr.softmax(logits, dim = 0)
    if length < 3:
        top_3 = tr.topk(probs, k=min(10, probs.numel()))
    else:
        top_3 = tr.topk(probs, k=min(3, probs.numel()))
    pro = [tokenizer.decode(item) for item in top_3.indices] # word
    orp = [item for item in top_3.values] # prob

    thing = zip(pro, orp)

    # print(top_3, len(top_3), logits.argmax(), pro)
    b = []
    for w, p in thing:
        new = Node(w, p)
        b.append(new)
    
    root.add_child(b)

    # for each generation, commit MCTS
    # score is prob
    for i in range(30):
        root.add_visits()
        s, r = MCTS(root)
    root = r
    if len(root.child) > 1:
        root.prune()

    if not done:
        for c in root.child:
            r = base(prompt + c.value, num_tokens, c, length + 1, root)
            c = r
    
    if len(root.child) > 1:
        root.prune()
    
    a = root
    b = a.child
    # word = a.value
    while len(b) > 1:
        a.prune()
        b = a.child
        
    return root
# ---------
def dfs(root:Node, max_depth, path, goal, prune):
    # path = []
    root.add_visits()
    if root.score == goal: 
        path.append(root.value)
        return path # find max P
    
    path.append(root.value)
    if len(root.child) == 0:
        return None
    for child in root.child:
        c_result = dfs(child, max_depth + 1, path, goal, prune)
        if c_result != None:
            return c_result
        c_result = path.pop(len(path)-1)
    if max_depth >= prune:
        root.prune(1)
    # root.prune(2)
    return None

def IDS(root, depth, goal, prune):
    result = None
    for max_depth in range(depth):
        result = dfs(root, max_depth, [], goal, prune)
        if result != None:
            break
    return result
        
        
def IDF_base(prompt, num_tokens, depth=0, root=None, base_root=2):
    # use same setup as evaluation
    tokenizer, model, mask = setup_llm()
    done = False
    leaf_nodes = []
    # initialize string to prompt before appending tokens
    string = prompt
    if not root:
        root = Node(string, 1)
    # generate child
    token_ids = tokenizer(string, return_tensors="pt").input_ids # (1, seq length)
    # done condition
    if token_ids.shape[1] == num_tokens: done = True

    # score the valid next tokens
    outputs = model(input_ids=token_ids)
    logits = outputs.logits[0,-1,:] # (vocab size,)
    logits = logits + mask

    # get probability table / policy
    probs = tr.log_softmax(logits, dim = 0)
    top_3 = tr.topk(probs, k=min(base_root, probs.numel()))

    # decode
    pro = [tokenizer.decode(item) for item in top_3.indices] # word
    orp = [item.item() for item in top_3.values] # prob

    thing = zip(pro, orp) # (word, prob)

    b = []
    goal_list = []
    for i, j in thing:
        j = j + root.prob
        new = Node(i, j)
        new.add_score(j)
        goal_list.append(j)
        b.append(new)
    
    goal = max(goal_list) # get the max prob
    root.add_child(b)

    # IDS(root, depth, goal)
    # v = [child.visits for child in root.child]
    # print(" ", depth, " ")
    test = []
    
    # make a couple generations
    # run IDS
    # prune
    # repeat
    gen = -100
    for child in root.child:
        # if len(root.child) <= 2:
        #     nice = True
        if not done: # and (child.visits != min(v) or depth % 2 == 0):
            de = depth + 1
            t, r, g, d, l = IDF_base(prompt+child.value, num_tokens, de, child)
            child = r
            leaf_nodes = leaf_nodes + l
            if type(t) == list:
                test += t
            else:
                test.append(t)
            
            if gen < g:
                gen = g
        else:
            leaf_nodes.append(child)
            test.append(prompt+child.value)
            if gen < goal:
                gen = goal
    
    return test, root, gen, depth+1, leaf_nodes

# the greedy approach like what was done in lecture, you can build off this if you want
def greedy(prompt, num_tokens):

    # use same setup as evaluation
    tokenizer, model, mask = setup_llm()

    # initialize string to prompt before appending tokens
    string = prompt

    # repeatedly append the most likely next token
    while True:

        # tokenize string so far
        token_ids = tokenizer(string, return_tensors="pt").input_ids # (1, seq length)

        # return once requested num tokens reached
        if token_ids.shape[1] == num_tokens: break

        # score the valid next tokens
        outputs = model(input_ids=token_ids)
        logits = outputs.logits[0,-1,:] # (vocab size,)
        logits = logits + mask

        # append the max score next token (greedy portion)
        max_token_id = logits.argmax()
        max_token = tokenizer.decode(max_token_id)
        string += max_token

    # return final result
    return string


### you must implement this function
# Before submission, make sure you use the search strategy that performed best, as this is what will be used for your code grade
def most_likely_string(prompt, num_tokens, option=0):
    # search algos
    # BFS, DFS, A*, MCTS, 
    if option == 0:
        token = 0
        g = None
        root = []
        d = [0]
        path = []

        new = Node(prompt, 1)
        test = Test(new)
        t = [(prompt, new)]
        num_tokens -= 1
        goal = 0
        pwe = 0
        while token <= num_tokens:
            if token + 2 >= num_tokens:
                token = num_tokens
                
            else:
                token += 2 # 2 more words
            
            next_layer = []
            same_gen = -100 # goal number
            d *= len(t)
            wow = []
            small = 0
            
            for w in range(len(t)): # for each prompt (leaf node)
                child = 2
                sequence, leaf_node = t[w]

                # return (paths), (tree), (goal), (depth), (leaf nodes)
                if leaf_node.score != goal: # or leaf_node.score == small:
                    child = 1
                a, r, g, depth, z = IDF_base(sequence, token, d[w], leaf_node, child) # tree
                wow.append(depth)

                test.attach_child(t[w][0], r)
                root.append(r)

                for s, n in zip(a,z):
                    if n.score < small:
                        small = n.score
                    next_layer.append((s, n)) # keep track of paths & nodes
                    # print((s,n.value, n.score))
                
                # print(g)
                if same_gen < g:
                    same_gen = g

                # print("\n___")
            # test.root.print(test.root.value)
            # prune the leaf nodes
            goal = same_gen
            remove = []
            count = 1
            if pwe % 2 == 0:
                count = 1
            for i in range(count):
                top = -100
                for m, n in next_layer:
                    if top < n.score:
                        top = n.score
                        remove.append((m,n))
                
                for item in remove:
                    if item in next_layer:
                        next_layer.remove(item)
            next_layer = remove

            t = next_layer
            pwe += 1
            # print(same_gen)
            d = [0]
            # d = wow
            path = IDS(test.root, num_tokens, g, token-1) # prunes in here
            # print(path)
            if token >= num_tokens:
                break
        path = IDS(test.root, num_tokens, g, num_tokens-1) # prunes in here

    # print(path)
    # print(len(t), d)

    # you can replace greedy with other strategies you investigate during experimental comparison
        return "".join(path) # greedy(prompt, num_tokens)
    elif option == 1:
        result = base(prompt, num_tokens)
        word = ""
        while len(result.child) > 0:
            word += result.value
            result = result.child[0]
        word += result.value
        return word
    else: 
        return greedy(prompt, num_tokens)


### you can use the main block for scratch work, informal testing, and running comparisons
if __name__ == "__main__":

    t = most_likely_string(prompt="  The", num_tokens=20)
    thing = ProbabilityEstimator()

    # print(t)
    # print(root.print(root.value))

    # word = ""
    # while len(root.child) > 0:
    #     word += root.value
    #     root = root.child[0]
    # word += root.value
    # print(word)
    # t = "  The following is an excerpt from the book, which was published in the United States in October, of this"
    print(t, "\n", thing.get_string_log_prob(t)[0])


