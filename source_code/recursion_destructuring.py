def direct_recursive_destructure(data: dict, 
                                 n_nestings: int=0,
                                 on_key_trail: bool=True,
                                 sep: str="-"
                                 ):
    """
    Recursively de-structure a JSON object and collect key-related information.

    Args:
        data (dict): The input JSON object to be de-structured.
        n_nestings (int): The number of nestings in the JSON structure.
        on_key_trail (bool): Whether to include key trails.
        sep (str): The separator used in key trails.

    Returns:
        tuple: A tuple containing the de-structured information, including keys, values, key trails, and nesting level.
    """
    if on_key_trail:
        trailed_keys = []
    keys_with_their_values = []
    only_keys = []
    nesting_found = False
    for key, value in data.items():
        # Only for nested dict-objects
        if isinstance(value, dict):
            if not nesting_found:
                n_nestings += 1
                nesting_found = True
            returned_only_keys, returned_keys_with_their_values, returned_trailed_keys, returned_n_nestings = direct_recursive_destructure(value,
                                                                                                                                         n_nestings)
            only_keys.append(key)
            keys_with_their_values.extend(returned_keys_with_their_values)
            if on_key_trail:
                temp = []
                for subkey in returned_trailed_keys:
                    if not isinstance(subkey, tuple):
                        temp.append(f"{key}{sep}{subkey}")
                    else:
                        sub1, sub2 = subkey
                        temp.append(f"{key}{sep}{sub1}{sep}{sub2}")
                trailed_keys.extend(tuple(temp))
            keys_with_their_values.append((key, tuple(returned_only_keys)))
        # Only for last dict-objects
        else:
            only_keys.append(key)
            keys_with_their_values.append((key, str(value)))
    if on_key_trail:
        return only_keys, keys_with_their_values, trailed_keys, n_nestings
    else:
        return only_keys, keys_with_their_values, returned_n_nestings
# only_keys, keys_with_their_values, trailed_keys, returned_n_nestings = direct_recursive_destructure(json_obj)

def indirect_recursive_json_destructure(data: dict, 
                                        on_key_trail: bool=True,
                                        sep: str="-"
                                        ):
    """
    Recursively de-structure a JSON object using an alternative approach.

    Args:
        data (dict): The input JSON object to be de-structured.
        on_key_trail (bool): Whether to include key trails.
        sep (str): The separator used in key trails.

    Returns:
        tuple: A tuple containing the de-structured information, including keys, values, key trails, and nesting level.
    """
    if on_key_trail:
        trailed_keys = []
    only_keys = []
    only_pairs = []
    keys_with_its_values = []
    ret_only_keys = []
    ret_keys_with_its_values = []
    ret_only_pairs = []
    k_iterators = []
    v_iterators = []
    
    keep_search = True
    saved = False
    next_node = True
    
    current_value = data.copy()
    # Iterators
    k_iterators = []
    v_iterators = []
    # Make iterators from level i-th keys
    k_iterator = iter(current_value.keys())
    v_iterator = iter(current_value.values())
    k_aligned = iter(current_value.keys())
    # Treat next level
    k_iterators.append(k_iterator)
    v_iterators.append(v_iterator)
    # Level counters
    max_level = 0
    curr_level = 1
    next_index = 0
    steps = 0

    while keep_search:
        # Try next key
        try: 
            if next_node:
                # Can go to StopIteration-1
                current_node = next(k_iterator)
                stored_key = current_node
                next_node = False
            # Can go to StopIteration-2
            current_value = next(v_iterator)
            # Won't fail. Previous next attempt to v_iterator will throw StopIteration first
            # because its length-aligned with k_iterator
            stored_key = next(k_aligned)
        except StopIteration as e:
            # If there's no more values from current node, we have to select the next node
            if not next_node:
                next_node = True
            # If there's no more nodes from current iterator, we have to select the previous iterator
            else:
                next_index = k_iterators.index(k_iterator)
                if next_index == 0:
                    keep_search = False
                    break
                if next_index >= 0:
                    k_iterator = k_iterators[next_index - 1]
                    v_iterator = v_iterators[next_index - 1]
                if curr_level > 1:
                    curr_level -= 1
        if isinstance(current_value, dict) and current_value:
            # Make iterators from level i-th keys
            k_iterator = iter(current_value.keys())
            v_iterator = iter(current_value.values())
            k_aligned = iter(current_value.keys())
            # Treat next level
            k_iterators.append(k_iterator)
            v_iterators.append(v_iterator)
            # Add up for new level
            curr_level += 1
        else:
            if curr_level > max_level:
                max_level = curr_level
        if not next_node:
            # Store data
            ret_only_keys.append(stored_key)
            if isinstance(current_value, (list, set)) and current_value:
                ret_keys_with_its_values.append((stored_key, tuple(current_value)))
            elif not isinstance(current_value, dict):
                ret_keys_with_its_values.append((stored_key, current_value))
                only_pairs.append((stored_key, current_value))
            else:
                ret_keys_with_its_values.append((stored_key, current_value))
            if saved:
                saved = False
        else:
            if not saved:
                only_keys.extend(ret_only_keys)
                trailed_keys.append(f"{sep}".join(ret_only_keys))
                keys_with_its_values.extend(ret_keys_with_its_values)
                only_pairs.extend(ret_only_pairs)
                ret_only_keys = []
                ret_keys_with_its_values = []
                ret_only_pairs = []
                saved = True
        steps += 1
    return only_keys, keys_with_its_values, trailed_keys, only_pairs, max_level, steps
