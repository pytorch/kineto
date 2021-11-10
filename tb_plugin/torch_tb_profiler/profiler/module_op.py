from copy import copy

from .node import ModuleNode, ProfilerStepNode
from .trace import EventTypes


def get_modules(tid2tree):
    modules = []

    def traverse_node(node, parent):
        if type(node) not in (ProfilerStepNode, ModuleNode):
            return

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

class Module:
    def __init__(self, name, module_id, python_id, python_parent_id):
        self.name = name
        self.module_id = module_id
        self.children = []
        self.python_id = python_id
        self.python_parent_id = python_parent_id

    def __hash__(self):
        if self.children:
            return hash((self.name, self.module_id, tuple(self.children)))
        else:
            return hash((self.name, self.module_id))

    def __eq__(self, o) -> bool:
        if not isinstance(o, Module):
            return False

        return (self.name == o.name and
                self.module_id == o.module_id and
                self.children == o.children)


def get_module_hierarchy(events):
    python_events = [e for e in events if e.type in (EventTypes.PYTHON_FUNCTION, EventTypes.MODULE)]
    python_events.sort(key=lambda e: e.python_id)
    id_to_event = {e.python_id: e for e in python_events}

    # Extract Python function topology.
    children = {}
    for e in python_events:
        e_id = e.python_id
        children.setdefault(e_id, [])
        e_parent_id = e.python_parent_id
        children.setdefault(e_parent_id, [])
        children[e_parent_id].append(e_id)
    function_leaves = [k for k, v in children.items() if not v]

    # Convert Python function topology to Module topology.
    #   This is a simple O(n) tree walking algorithm where we start from the leaves
    #   and walk up, discarding any nodes which are not Module nodes.
    module_parent_map = {}
    seen = set()
    for i in function_leaves:
        e = id_to_event[i]
        current_module = None
        while e is not None:
            e_id = e.python_id
            if e.type == EventTypes.MODULE:
                if current_module is not None:
                    module_parent_map[current_module.python_id] = e_id
                current_module = e
                module_parent_map.setdefault(e_id, None)

            seen_key = (e_id, id(current_module))
            if seen_key in seen:
                break
            seen.add(seen_key)

            e = id_to_event.get(e.python_parent_id, None)

    module_roots = [k for k, v in module_parent_map.items() if v is None]
    module_child_map = {}
    for child_id, parent_id in module_parent_map.items():
        module_child_map.setdefault(child_id, [])
        module_child_map.setdefault(parent_id, [])
        module_child_map[parent_id].append(child_id)

    # The traverse order is well defined which guarantees that a given topology
    # will produce a unique and unambiguous hierarchy.
    def append_hierarchy(e_id):
        e = id_to_event[e_id]
        module = Module(e.name, e.module_id, e.python_id, e.python_parent_id)
        for id in module_child_map[e_id]:
            child = append_hierarchy(id)
            module.children.append(child)
        return module

    unique_modules = set()
    for e_id in module_roots:
        root = append_hierarchy(e_id)
        unique_modules.add(root)

    return list(unique_modules)

def dump_modules(level, modules):
    for module in modules:
        print(f"{'    ' * level}{module.name.replace('nn.Module: ', '')}_{module.module_id}")
        dump_modules(level + 1, module.children)
