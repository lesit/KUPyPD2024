import os
import ast
import json

from anytree import Node
from anytree.search import findall_by_attr
from anytree.walker import Walker

def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'  # node.pop()
    elif isinstance(node, ast.AST):
        token = node.__class__.__name__

    return token

def get_children(root):
    if isinstance(root, ast.AST):
        children = ast.iter_child_nodes(root)
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))

def get_trees(current_node, parent_node, order):
    
    token, children = get_token(current_node), get_children(current_node)
    node = Node([order,token], parent=parent_node, order=order)

    for child_order, child in enumerate(children):
        get_trees(child, node, order+str(int(child_order)+1))

def get_path_length(path):
    """Calculating path length.
    Input:
    path: list. Containing full walk path.

    Return:
    int. Length of the path.
    """
    
    return len(path)

def get_path_width(raw_path):
    """Calculating path width.
    Input:
    raw_path: tuple. Containing upstream, parent, downstream of the path.

    Return:
    int. Width of the path.
    """
    
    """
    raw_path[0][-1] : upper node of start
    raw_path[2][0] : upper node of end
    therefore, levels of two nodes are same
    """
    
    return abs(int(raw_path[0][-1].order)-int(raw_path[2][0].order))
    
def get_node_rank(node_name, max_depth):
    """Calculating node rank for leaf nodes.
    Input:
    node_name: list. where the first element is the string order of the node, second element is actual name.
    max_depth: int. the max depth of the code.

    Return:
    list. updated node name list.
    """
    while len(node_name[0]) < max_depth:
        node_name[0] += "0"
    return [int(node_name[0]),node_name[1]]


def pycode_extract_path(code, max_length, max_width):
    """Extracting paths for a given json code.
    Input:
    json_code: json object. The json object of a snap program to be extracted.
    max_length: int. Max length of the path to be restained.
    max_width: int. Max width of the path to be restained.
    hashing_table: Dict. Hashing table for path. if not None, MD5 hashed path will be returned to save space.

    Return:
    walk_paths: list of AST paths from the json code.
    """
    
    # Initialize head node of the code.
    head = Node(["1",get_token(code)])
    
    # Recursively construct AST tree.
    
    for child_order, child in enumerate(get_children(code)):
        get_trees(child, head, "1"+str(int(child_order)+1))
    
    # Getting leaf nodes.
    leaf_nodes = findall_by_attr(head, name="is_leaf", value=True)
    
    # Getting max depth.
    max_depth = max([len(node.name[0]) for node in leaf_nodes])
    
    # Node rank modification.
    for leaf in leaf_nodes:
        leaf.name = get_node_rank(leaf.name,max_depth)
    
    walker = Walker()
    text_paths = []
    path_length_list = []
    # Walk from leaf to target
    for leaf_index in range(len(leaf_nodes)-1):
        for target_index in range(leaf_index+1, len(leaf_nodes)):
            raw_path = walker.walk(leaf_nodes[leaf_index], leaf_nodes[target_index])
            if get_path_width(raw_path) > max_width:
                continue
            
            # Combining up and down streams
            walk_path = [n.name[1] for n in list(raw_path[0])]+[raw_path[1].name[1]]+[n.name[1] for n in list(raw_path[2])]
            for node in walk_path:
                assert node.find(",")<0

            text_path = "_".join(walk_path)
            
            # Only keeping satisfying paths.
            path_len = get_path_length(walk_path)
            path_length_list.append(path_len)

            if path_len <= max_length:
                text_paths.append(walk_path[0]+","+text_path+","+walk_path[-1])
    if len(text_paths) == 0:
        a = 0
    return text_paths, path_length_list

import sys
import time

module_path = os.path.abspath("../..")
if module_path not in sys.path:
    sys.path.append(module_path)

import src.make_logger as make_logger
from src.preprocess_ipython_uncompiled import *

def path_extract(code_df, config, logger, logger_name, log_dir):
    parsed_code_dict = dict()
    uncompiled_list = []
    for idx, row in code_df.iterrows():
        code = row["code"]
        try:
            parsed_code = ast.parse(code)
            # logger.info(code)
        except Exception as e:
            parsed_code = "Uncompilable"
            uncompiled_list.append(code)
        parsed_code_dict[row.code_id] = (row.student_id, parsed_code)

    with open("uncompiled.json", "w") as f:
        json.dump(uncompiled_list, f, indent=3)
    logger.info(f"parsed: {len(parsed_code_dict)}. compiled: {len(parsed_code_dict)-len(uncompiled_list)}. uncompiled: {len(uncompiled_list)}")

    def extract_process(mp_idx, shared_result_path_dict, shared_count_save_dict, shared_lock):
        st = time.time()

        mp_logger = make_logger.make(logger_name+f"_mp{mp_idx}", time_filename=False, save_dir=log_dir)
        mp_logger.info("start")

        total_count = 0
        for idx, (code_id, (student_id, parsed_code)) in enumerate(parsed_code_dict.items()):
            shared_lock.acquire()

            if code_id in shared_result_path_dict:
                # print(f"{idx} already")
                shared_lock.release()
                continue

            shared_result_path_dict[code_id] = None      # preempt
            shared_lock.release()

            ast_paths, code_path_length_list = pycode_extract_path(parsed_code, 
                                                                    max_length=config.code_path_length, 
                                                                    max_width=config.code_path_width)
            shared_result_path_dict[code_id] = [student_id, "@".join(ast_paths), len(ast_paths), code_path_length_list]  # save result
            total_count += 1

            if (idx+1) % 100 == 0:
                mp_logger.info(f"{idx+1}/{len(parsed_code_dict)}. extract")

        shared_count_save_dict[mp_idx] = total_count
        mp_logger.info(f"end. total_processed:{total_count} elapse:{time.time()-st}")

    cores = min(4, os.cpu_count())
    # cores = 1

    import multiprocessing
    shared_lock = multiprocessing.Lock()
    manager = multiprocessing.Manager()
    shared_result_path_dict = manager.dict()
    shared_count_save_dict = manager.dict()

    process_list = []
    for idx in range(cores):
        process = multiprocessing.Process(target=extract_process, args=[idx, shared_result_path_dict, shared_count_save_dict, shared_lock])
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()

    raw_paths_list = [[v[0], code_id, v[1]] for code_id, v in shared_result_path_dict.items()]
    assert len(raw_paths_list) == len(parsed_code_dict)

    using_n_path_list = [x[2] for x in shared_result_path_dict.values()]
    code_path_length_list = [x[3] for x in shared_result_path_dict.values()]

    return raw_paths_list, using_n_path_list, code_path_length_list
