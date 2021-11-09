from copy import copy

from .node import ModuleNode

def get_module_hierarchy(tid2tree):
    modules = []

    def traverse_node(node, parent):
        if isinstance(node, ModuleNode):
            module = copy(node)
            # remove the children after copy to keep the module only
            module.children = []

            if parent is None:
                modules.append(module)
            else:
                parent.children.append(module)
            parent = module

        for child in node.children:
            traverse_node(child, parent)

    for _, root in tid2tree.items():
        for child in root.children:
            # since the root node is CallTreeRoot, there is no parent ModuleNode
            traverse_node(child, None)

    return modules


def dump_modules(level, modules):
    for module in modules:
        print("    " * level, module.name)
        dump_modules(level + 1, module.children)
