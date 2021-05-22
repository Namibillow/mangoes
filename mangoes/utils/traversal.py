"""Helper functions for the dependency based context.
"""

def find_parents(count, depth, idx, label, sentence_tree, parents, visited):
    """Find all the connecting element from given starting element.

    Parameters
    -----------
    count: int. Denotes how far away from current node.
    depth: int. Denotes maximal depth.
    idx: int
        index of word in a sentence.
    label: str. Contains dependency relation.
    sentence_tree: list
    parents: set. Append all nodes that are reachable for idx. 
    visited: set. 
    """
    stack = [(idx, label, count)]

    while stack:
        idx, label, count = stack.pop()
        # print(idx, label, count)
        if count > depth:
            return 

        parents.add((idx, label, count))
        for token, token_children in enumerate(sentence_tree):
            for child, child_label in token_children:
                if child == idx and idx not in visited:
                    visited.add(idx)
                    new_label = label + "+" + child_label if label else child_label
                    stack.append((token, new_label, count+1))
   

def dfs(remain, node, label, visited, new_children, sentence_tree):
    """Find path that connects current element up to depth-far.

    Parameters
    -----------
    remain: int. How far node still can travarse. 
    node: int 
        index of word in a sentence.
    label: str. Contains dependency relation.
    visited: set 
    new_children: set 
    sentence_tree: list.
    """
    if remain < 0:
        return 
    
    stack = [(remain, node, label)]

    while stack:
        remain, node, label = stack.pop()

        if remain < 0:
            continue 
        
        if label:
            new_children.add((node, label))

        for j, token_children in enumerate(sentence_tree):
            if j not in visited:
                for child, child_label in token_children:
                    if child == node:
                        visited.add(node)
                        new_label = label+ "+" + child_label if label else child_label
                        stack.append((remain-1, j, new_label))
