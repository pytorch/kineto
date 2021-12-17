# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
from collections import namedtuple
from typing import Dict, Generator, Iterable, List, Optional, Set, Tuple, Union

from .node import ModuleNode, OperatorNode, ProfilerStepNode, is_operator_node
from .trace import BaseEvent, EventTypes, PythonFunctionEvent


class Module:
    def __init__(self, name: str, module_id: int):
        self.name = name
        self.module_id = module_id
        self.children: List[Module] = []

    def __hash__(self):
        return hash((self.name, self.module_id, tuple(self.children)))

    def __eq__(self, o) -> bool:
        if not isinstance(o, Module):
            return False

        return (self.name == o.name and
                self.module_id == o.module_id and
                self.children == o.children)


class ModuleStats:
    def __init__(self, name: str, module_id: int):
        self.name = name
        self.module_id = module_id
        self.occurences: int = 0
        self.operators: int = 0
        self.host_duration: int = 0
        self.device_duration: int = 0
        self.self_host_duration: int = 0
        self.self_device_duration: int = 0

    @property
    def avg_host_duration(self):
        return self.host_duration / self.occurences

    @property
    def avg_device_duration(self):
        return self.device_duration / self.occurences


Stats = namedtuple('Stats', [
    'name',
    'id',
    'occurences',
    'operators',
    'host_duration',
    'self_host_duration',
    'device_duration',
    'self_device_duration',
    'avg_duration',
    'children'])


def aggegate_module_view(tid2tree: Dict[int, OperatorNode], events: List[BaseEvent]) -> Optional[List[Stats]]:
    roots = _build_module_hierarchy(events)
    modules = _get_module_list(tid2tree)
    if modules and roots:
        return _process_module_statistics(modules, roots)
    else:
        return None


def _build_module_hierarchy(events: List[PythonFunctionEvent]) -> List[Module]:
    """Get the module hierarchy from the chome trace events
    """
    python_events = [e for e in events if e.type in (EventTypes.PYTHON_FUNCTION, EventTypes.MODULE)]
    id_to_event = {e.python_id: e for e in python_events}

    # Extract Python function topology.
    children: Dict[int, List[int]] = {}
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
    module_child_map: Dict[int, List[int]] = {}
    for child_id, parent_id in module_parent_map.items():
        module_child_map.setdefault(child_id, [])
        module_child_map.setdefault(parent_id, [])
        module_child_map[parent_id].append(child_id)

    # The traverse order is well defined which guarantees that a given topology
    # will produce a unique and unambiguous hierarchy.
    def append_hierarchy(e_id) -> Module:
        e = id_to_event[e_id]
        module = Module(e.name, e.module_id)
        for id in module_child_map[e_id]:
            child = append_hierarchy(id)
            module.children.append(child)
        return module

    unique_modules: Set[Module] = set()
    for e_id in module_roots:
        root = append_hierarchy(e_id)
        unique_modules.add(root)

    return list(unique_modules)


def _aggregate_modules(modules: Iterable[ModuleNode]) -> Dict[Tuple[str, int], ModuleStats]:
    """Aggregate the modules based on the name and module_id"""
    module_aggs: Dict[Tuple(str, int), ModuleStats] = {}
    for m in modules:
        key = (m.name, m.module_id)
        if key not in module_aggs:
            module_aggs[key] = ModuleStats(m.name, m.module_id)
        agg = module_aggs[key]
        agg.occurences += 1

        agg.operators += sum(is_operator_node(child) for child in m.children)

        agg.self_host_duration += m.self_host_duration
        agg.host_duration += m.end_time - m.start_time

        agg.self_device_duration += m.self_device_duration
        agg.device_duration += m.device_duration

    return module_aggs


def _get_module_list(tid2tree: Dict[int, OperatorNode]) -> Generator[ModuleNode, None, None]:
    """Get all ModuleNode from the operator tree"""
    def traverse_node(node):
        if type(node) not in (ProfilerStepNode, ModuleNode, OperatorNode):
            return

        if isinstance(node, ModuleNode):
            yield node

        for child in node.children:
            yield from traverse_node(child)

    for _, root in tid2tree.items():
        for child in root.children:
            yield from traverse_node(child)


def _process_module_statistics(modules_nodes: Iterable[ModuleNode], hierarchy: Iterable[Module]) -> List[Stats]:
    """Get the module statistics from the ModuleNode(s) and the hierarchy
    """
    module_aggs = _aggregate_modules(modules_nodes)

    def process_modules(h_modules: Iterable[Module]):
        for m in h_modules:
            name = m.name.replace('nn.Module: ', '')
            stats = module_aggs[(m.name, m.module_id)]

            child_stats = list(process_modules(m.children))
            yield Stats(
                name,
                m.module_id,
                stats.occurences,
                stats.operators,
                stats.host_duration,
                stats.self_host_duration,
                stats.device_duration,
                stats.self_device_duration,
                stats.avg_device_duration if stats.avg_device_duration > 0 else stats.avg_host_duration,
                child_stats)

    data = sorted(process_modules(hierarchy), key=lambda x: x.name)
    return data


def get_module_tree(tid2tree: Dict[int, OperatorNode]):
    """Get the module tree in timeline"""
    from copy import copy

    modules = []

    def traverse_node(node, parent: Optional[ModuleNode]):
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


def dump_modules(level: int, modules: Iterable[Union[Module, ModuleNode]]):
    """testing purpose"""
    for module in modules:
        print(f"{'    ' * level}{module.name.replace('nn.Module: ', '')}_{module.module_id}")
        dump_modules(level + 1, module.children)
