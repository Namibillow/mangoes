"""Helper functions for the dependency based context.
"""

def find_parents(count, depth, idx, label, sentence_tree, parents, visited):

    if count > depth:
        return 

    if idx == 0:
        parents.add((idx, "", count))
    else:
        parents.add((idx, label, count))

    visited.add(idx)
    for token, token_children in enumerate(sentence_tree):
        for child, child_label in token_children:
                if child == idx:
                    new_label = label + "+" + child_label if label else child_label
                    find_parents(count+1, depth, token, new_label, sentence_tree, parents, visited)
        

def dfs(remain, node, label,  visited, new_children, sentence_tree):
    if remain < 0:
        return 

    if label:
        new_children.add((node, label))

    visited.add(node)

    for j, token_children in enumerate(sentence_tree):
        if j not in visited:
            for child, child_label in token_children:
                if child == node:
                    new_label = label+ "+" + child_label if label else child_label
                    dfs(remain-1, j, new_label, visited, new_children, sentence_tree)
    
    