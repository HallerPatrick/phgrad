def string_format_module_tree(module_tree, indent=0):
    """Format the module tree into a string."""
    module_tree_str = ""
    for name, submodule in module_tree.items():
        module_tree_str += "  " * indent + f"{name}:\n"
        if isinstance(submodule, dict):
            module_tree_str += string_format_module_tree(submodule, indent + 1)
        else:
            module_tree_str += "  " * (indent + 1) + f"{submodule}\n"
    return module_tree_str
