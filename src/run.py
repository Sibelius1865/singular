import random
import copy
import ast
from typing import List, Dict, Callable, Optional
from datetime import datetime
import os

import torch
from torch_geometric.data import Data
import networkx as nx

try:
    from .node_type import NodeType
    from .visualize import visualize_graph
except ImportError:
    from node_type import NodeType
    from visualize import visualize_graph


# =========================
# Operation definitions
# =========================

OPS = {
    "id": lambda x: x,
    "add1": lambda x: x + 1,
    "mul2": lambda x: x * 2,
    "square": lambda x: x * x,
}

OP_NAMES = list(OPS.keys())


# =========================
# Graph IR definition
# =========================

class GraphIR:
    """
    Minimal machine-friendly Graph IR
    """

    def __init__(
        self,
        node_features: torch.Tensor,  # [N, F]
        node_types: torch.Tensor,     # [N]  (NodeType)
        edge_index: torch.Tensor,      # [2, E]
        edge_attr: torch.Tensor = None, # [E, A]
        node_ops: Dict[int, str] = None  # {node_id: op_name} for COMPUTE nodes
    ):
        self.node_features = node_features
        self.node_types = node_types
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.node_ops = node_ops if node_ops is not None else {}

    @property
    def num_nodes(self):
        return self.node_features.size(0)
    
    def copy(self):
        """Create a deep copy of the GraphIR"""
        return GraphIR(
            node_features=self.node_features.clone(),
            node_types=self.node_types.clone(),
            edge_index=self.edge_index.clone(),
            edge_attr=self.edge_attr.clone() if self.edge_attr is not None else None,
            node_ops=self.node_ops.copy()
        )


# =========================
# IR <-> PyG conversion
# =========================

def ir_to_pyg(ir: GraphIR) -> Data:
    """
    Convert GraphIR to PyTorch Geometric Data
    """
    return Data(
        x=ir.node_features,
        edge_index=ir.edge_index,
        edge_attr=ir.edge_attr,
        node_type=ir.node_types  # include node types
    )


# =========================
# Graph execution
# =========================

def execute_graph(ir: GraphIR, input_value: float) -> float:
    """
    Execute GraphIR as a pure function.
    
    Args:
        ir: GraphIR to execute
        input_value: Input value for INPUT nodes
    
    Returns:
        Output value from OUTPUT node
    
    Raises:
        ValueError: If graph structure is invalid
    """
    g = _ir_to_nx(ir)
    env = {}
    
    # Topological sort to ensure dependencies are computed first
    try:
        order = list(nx.topological_sort(g))
    except nx.NetworkXError:
        raise ValueError("Graph contains cycles, cannot execute")
    
    for node in order:
        ntype = ir.node_types[node].item()
        
        if ntype == NodeType.INPUT:
            env[node] = input_value
        
        elif ntype == NodeType.COMPUTE:
            preds = list(g.predecessors(node))
            if len(preds) == 0:
                raise ValueError(f"COMPUTE node {node} has no inputs")
            elif len(preds) == 1:
                x = env[preds[0]]
            else:
                # Multiple inputs: use the first one (simple strategy for unary ops)
                # In future, could implement merging strategies (sum, max, etc.)
                # for binary/multi-ary operations
                x = env[preds[0]]
            
            if node not in ir.node_ops:
                raise ValueError(
                    f"COMPUTE node {node} has no operation assigned"
                )
            
            op_name = ir.node_ops[node]
            if op_name not in OPS:
                raise ValueError(f"Unknown operation: {op_name}")
            
            env[node] = OPS[op_name](x)
        
        elif ntype == NodeType.OUTPUT:
            preds = list(g.predecessors(node))
            if len(preds) == 0:
                raise ValueError(f"OUTPUT node {node} has no inputs")
            elif len(preds) == 1:
                env[node] = env[preds[0]]
            else:
                # Multiple inputs: use the first one (simple strategy)
                # In future, could implement merging strategies (sum, max, etc.)
                env[node] = env[preds[0]]
        
        else:
            raise ValueError(f"Unknown node type: {ntype}")
    
    # Find OUTPUT node and return its value
    output_nodes = [
        n for n in range(ir.num_nodes)
        if ir.node_types[n].item() == NodeType.OUTPUT
    ]
    
    if len(output_nodes) != 1:
        raise ValueError(
            f"Graph must have exactly one OUTPUT node (got {len(output_nodes)})"
        )
    
    return env[output_nodes[0]]


# =========================
# GraphIR → Python AST conversion
# =========================

def graph_to_ast(graph: GraphIR) -> ast.Module:
    """
    Convert GraphIR to Python AST.
    
    Generates a function of the form:
    def generated_fn(x):
        n0 = x
        n1 = OPS["mul2"](n0)
        n2 = OPS["add1"](n1)
        return n2
    
    Args:
        graph: GraphIR to convert
    
    Returns:
        ast.Module containing the function definition
    """
    g = _ir_to_nx(graph)
    
    # Get topological order
    try:
        order = list(nx.topological_sort(g))
    except nx.NetworkXError:
        raise ValueError("Graph contains cycles, cannot convert to AST")
    
    # Find INPUT and OUTPUT nodes
    input_nodes = [
        n for n in range(graph.num_nodes)
        if graph.node_types[n].item() == NodeType.INPUT
    ]
    output_nodes = [
        n for n in range(graph.num_nodes)
        if graph.node_types[n].item() == NodeType.OUTPUT
    ]
    
    if len(input_nodes) != 1:
        raise ValueError(f"Graph must have exactly one INPUT node (got {len(input_nodes)})")
    if len(output_nodes) != 1:
        raise ValueError(f"Graph must have exactly one OUTPUT node (got {len(output_nodes)})")
    
    input_node = input_nodes[0]
    output_node = output_nodes[0]
    
    # Build function body statements
    body = []
    
    # Process nodes in topological order
    for node in order:
        ntype = graph.node_types[node].item()
        
        if ntype == NodeType.INPUT:
            # INPUT node: bind to function argument x
            # n{node} = x
            body.append(
                ast.Assign(
                    targets=[ast.Name(id=f"n{node}", ctx=ast.Store())],
                    value=ast.Name(id="x", ctx=ast.Load())
                )
            )
        
        elif ntype == NodeType.COMPUTE:
            # COMPUTE node: apply operation to predecessor
            preds = list(g.predecessors(node))
            if len(preds) == 0:
                raise ValueError(f"COMPUTE node {node} has no inputs")
            
            # Use first predecessor (same logic as execute_graph)
            pred_node = preds[0]
            pred_var = ast.Name(id=f"n{pred_node}", ctx=ast.Load())
            
            if node not in graph.node_ops:
                raise ValueError(f"COMPUTE node {node} has no operation assigned")
            
            op_name = graph.node_ops[node]
            
            # Build: n{node} = OPS["op_name"](n{pred_node})
            body.append(
                ast.Assign(
                    targets=[ast.Name(id=f"n{node}", ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Subscript(
                            value=ast.Name(id="OPS", ctx=ast.Load()),
                            slice=ast.Constant(value=op_name),
                            ctx=ast.Load()
                        ),
                        args=[pred_var],
                        keywords=[]
                    )
                )
            )
        
        elif ntype == NodeType.OUTPUT:
            # OUTPUT node: return predecessor value
            preds = list(g.predecessors(node))
            if len(preds) == 0:
                raise ValueError(f"OUTPUT node {node} has no inputs")
            
            # Use first predecessor (same logic as execute_graph)
            pred_node = preds[0]
            pred_var = ast.Name(id=f"n{pred_node}", ctx=ast.Load())
            
            # Build: return n{pred_node}
            body.append(ast.Return(value=pred_var))
    
    # Create function definition
    func_def = ast.FunctionDef(
        name="generated_fn",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="x", annotation=None)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        ),
        body=body,
        decorator_list=[],
        returns=None
    )
    
    # Create module
    module = ast.Module(body=[func_def], type_ignores=[])
    
    # Fix missing locations (required for compilation)
    ast.fix_missing_locations(module)
    
    return module


def compile_graph(graph: GraphIR) -> Callable:
    """
    Compile GraphIR to a callable Python function.
    
    Args:
        graph: GraphIR to compile
    
    Returns:
        Callable function that takes input_value and returns output
    """
    ast_module = graph_to_ast(graph)
    
    # Compile AST to code object
    code = compile(ast_module, filename="<generated>", mode="exec")
    
    # Execute in a namespace that includes OPS
    namespace = {"OPS": OPS}
    exec(code, namespace)
    
    # Extract the generated function
    if "generated_fn" not in namespace:
        raise RuntimeError("Failed to generate function from AST")
    
    return namespace["generated_fn"]


def ast_to_source(ast_module: ast.Module) -> str:
    """
    Convert AST module to Python source code string.
    
    Args:
        ast_module: AST module to convert
    
    Returns:
        Python source code as string
    """
    try:
        # Python 3.9+ has ast.unparse
        if hasattr(ast, 'unparse'):
            return ast.unparse(ast_module)
        else:
            # Fallback for older Python versions
            # Use astor or manual string building
            # For now, raise informative error
            raise NotImplementedError(
                "ast.unparse is not available (Python < 3.9). "
                "Please use Python 3.9+ or install astor package."
            )
    except AttributeError:
        raise NotImplementedError(
            "ast.unparse is not available. Please use Python 3.9+."
        )


def save_graph_as_function(graph: GraphIR, filepath: str, function_name: str = "generated_fn"):
    """
    Save GraphIR as a Python function to a file.
    
    Args:
        graph: GraphIR to save
        filepath: Path to save the Python file
        function_name: Name for the generated function
    """
    ast_module = graph_to_ast(graph)
    
    # Rename function if needed
    if function_name != "generated_fn":
        for node in ast.walk(ast_module):
            if isinstance(node, ast.FunctionDef) and node.name == "generated_fn":
                node.name = function_name
    
    source = ast_to_source(ast_module)
    
    # Add OPS import at the top
    full_source = f"""# Auto-generated from GraphIR
# This file was generated automatically. Do not edit manually.

from run import OPS

{source}
"""
    
    with open(filepath, 'w') as f:
        f.write(full_source)


# =========================
# Semantic path checking
# =========================

def _ir_to_nx(ir: GraphIR) -> nx.DiGraph:
    """
    Convert GraphIR to NetworkX DiGraph
    """
    g = nx.DiGraph()
    g.add_nodes_from(range(ir.num_nodes))
    g.add_edges_from(ir.edge_index.t().tolist())
    return g


def has_semantic_path(ir: GraphIR) -> bool:
    """
    Check if there exists at least one INPUT -> OUTPUT path
    """
    g = _ir_to_nx(ir)
    inputs = (ir.node_types == NodeType.INPUT).nonzero(as_tuple=True)[0]
    outputs = (ir.node_types == NodeType.OUTPUT).nonzero(as_tuple=True)[0]

    for i in inputs.tolist():
        for o in outputs.tolist():
            if nx.has_path(g, i, o):
                return True

    return False


def semantic_paths(ir: GraphIR) -> List[List[int]]:
    """
    Get all simple paths from INPUT nodes to OUTPUT nodes
    Returns list of paths (each path is a list of node indices)
    """
    g = _ir_to_nx(ir)
    inputs = (ir.node_types == NodeType.INPUT).nonzero(as_tuple=True)[0].tolist()
    outputs = (ir.node_types == NodeType.OUTPUT).nonzero(as_tuple=True)[0].tolist()

    all_paths = []
    for i in inputs:
        for o in outputs:
            try:
                paths = list(nx.all_simple_paths(g, i, o))
                all_paths.extend(paths)
            except nx.NetworkXNoPath:
                continue

    return all_paths


def count_compute_in_path(path: List[int], ir: GraphIR) -> int:
    """
    Count COMPUTE nodes in a path
    """
    return sum(1 for n in path if ir.node_types[n] == NodeType.COMPUTE)


def get_node_degrees(ir: GraphIR) -> tuple:
    """
    Get in-degree and out-degree for each node
    Returns (in_degrees, out_degrees) as dicts
    """
    g = _ir_to_nx(ir)
    in_degrees = dict(g.in_degree())
    out_degrees = dict(g.out_degree())
    return in_degrees, out_degrees


def count_dead_compute(ir: GraphIR) -> int:
    """
    Count COMPUTE nodes that cannot reach any OUTPUT
    """
    g = _ir_to_nx(ir)
    outputs = (ir.node_types == NodeType.OUTPUT).nonzero(as_tuple=True)[0].tolist()
    
    dead_count = 0
    for node_id in range(ir.num_nodes):
        if ir.node_types[node_id] == NodeType.COMPUTE:
            can_reach_output = any(
                nx.has_path(g, node_id, o) for o in outputs
            )
            if not can_reach_output:
                dead_count += 1
    
    return dead_count


# =========================
# Semantic fitness function
# =========================

def fitness(ir: GraphIR) -> float:
    """
    Semantic fitness function that evaluates:
    (A) Number of semantic paths (diversity)
    (B) Average compute depth per path
    (C) Branch/merge structure (nodes with indegree>=2 or outdegree>=2)
    (D) Dead compute penalty (COMPUTE nodes that don't reach OUTPUT)
    (E) Regularization (size penalty)
    """
    paths = semantic_paths(ir)
    
    if not paths:
        return -1e9  # Insurance (should not happen with semantic_mutate)
    
    # (A) Path count score (diversity)
    # Cap at reasonable number to avoid explosion
    # Use log scale for very large path counts to prevent dominance
    path_count = len(paths)
    if path_count <= 20:
        path_score = path_count
    else:
        # Logarithmic scaling for paths > 20
        path_score = 20 + 2 * (path_count - 20) ** 0.5
    
    # (B) Average compute depth per path
    total_compute_in_paths = sum(count_compute_in_path(p, ir) for p in paths)
    depth_score = total_compute_in_paths / len(paths)
    
    # (C) Branch/merge structure score
    in_degrees, out_degrees = get_node_degrees(ir)
    branch_score = sum(
        1 for n in range(ir.num_nodes)
        if ir.node_types[n] == NodeType.COMPUTE
        and (in_degrees.get(n, 0) >= 2 or out_degrees.get(n, 0) >= 2)
    )
    
    # (D) Dead compute penalty
    dead_nodes = count_dead_compute(ir)
    
    # (E) Size regularization
    size_penalty = 0.01 * ir.num_nodes + 0.005 * ir.edge_index.size(1)
    
    # Weighted combination
    return (
        1.0 * path_score +
        0.5 * depth_score +
        0.3 * branch_score -
        0.2 * dead_nodes -
        size_penalty
    )


# =========================
# Semantic-preserving mutation operators
# =========================

def mutate_insert_compute(ir: GraphIR, feature_dim: int) -> GraphIR:
    """
    Insert a COMPUTE node in the middle of an existing edge
    A -> B  =>  A -> C -> B
    """
    if ir.edge_index.size(1) == 0:
        return ir

    new_ir = ir.copy()

    # pick an existing edge
    idx = random.randint(0, new_ir.edge_index.size(1) - 1)
    src, dst = new_ir.edge_index[:, idx].tolist()

    # new compute node
    new_node_id = new_ir.num_nodes

    new_ir.node_features = torch.cat(
        [new_ir.node_features, torch.randn(1, feature_dim)], dim=0
    )
    new_ir.node_types = torch.cat(
        [new_ir.node_types, torch.tensor([NodeType.COMPUTE])], dim=0
    )
    
    # Assign random operation to new COMPUTE node
    new_ir.node_ops[new_node_id] = random.choice(OP_NAMES)

    # remove old edge
    mask = torch.ones(new_ir.edge_index.size(1), dtype=torch.bool)
    mask[idx] = False
    new_ir.edge_index = new_ir.edge_index[:, mask]

    # add new edges
    new_edges = torch.tensor(
        [[src, new_node_id], [new_node_id, dst]],
        dtype=torch.long
    ).t()

    new_ir.edge_index = torch.cat(
        [new_ir.edge_index, new_edges], dim=1
    )

    return new_ir


def mutate_parallel_compute(ir: GraphIR, feature_dim: int) -> GraphIR:
    """
    Add a parallel COMPUTE path
    A -> B  =>  A -> C -> B (parallel to A -> B)
    """
    if ir.edge_index.size(1) == 0:
        return ir

    new_ir = ir.copy()

    idx = random.randint(0, new_ir.edge_index.size(1) - 1)
    src, dst = new_ir.edge_index[:, idx].tolist()

    new_node_id = new_ir.num_nodes

    new_ir.node_features = torch.cat(
        [new_ir.node_features, torch.randn(1, feature_dim)], dim=0
    )
    new_ir.node_types = torch.cat(
        [new_ir.node_types, torch.tensor([NodeType.COMPUTE])], dim=0
    )
    
    # Assign random operation to new COMPUTE node
    new_ir.node_ops[new_node_id] = random.choice(OP_NAMES)

    new_edges = torch.tensor(
        [[src, new_node_id], [new_node_id, dst]],
        dtype=torch.long
    ).t()

    new_ir.edge_index = torch.cat(
        [new_ir.edge_index, new_edges], dim=1
    )

    return new_ir


def semantic_mutate(ir: GraphIR, feature_dim: int) -> GraphIR:
    """
    Semantic-preserving mutation with retry mechanism
    """
    for _ in range(5):  # safety retry
        if random.random() < 0.5:
            candidate = mutate_insert_compute(ir, feature_dim)
        else:
            candidate = mutate_parallel_compute(ir, feature_dim)

        if has_semantic_path(candidate):
            return candidate

    return ir.copy()  # fallback (no-op)


# =========================
# Logging utilities
# =========================

_log_file: Optional[object] = None


def setup_logging(log_dir: str = "logs") -> str:
    """
    Setup logging to a file with timestamp.
    
    Args:
        log_dir: Directory to save log files
    
    Returns:
        Path to the log file
    """
    global _log_file
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{timestamp}.log")
    
    _log_file = open(log_path, 'w', encoding='utf-8')
    
    # Write header
    _log_file.write(f"=== Graph IR Evolution Run ===\n")
    _log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _log_file.write(f"{'='*50}\n\n")
    
    return log_path


def log_write(message: str, also_print: bool = True):
    """
    Write message to log file and optionally print to console.
    
    Args:
        message: Message to write
        also_print: Whether to also print to console
    """
    global _log_file
    
    if also_print:
        print(message)
    
    if _log_file is not None:
        _log_file.write(message + '\n')
        _log_file.flush()  # Ensure immediate write


def close_logging():
    """Close the log file."""
    global _log_file
    
    if _log_file is not None:
        _log_file.write(f"\n{'='*50}\n")
        _log_file.write(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        _log_file.close()
        _log_file = None


# =========================
# Genetic Algorithm loop
# =========================

def evolve(
    population: List[GraphIR],
    generations: int,
    feature_dim: int,
    elite_ratio: float = 0.3
) -> List[GraphIR]:

    for gen in range(generations):
        scored = [(fitness(ir), ir) for ir in population]
        scored.sort(key=lambda x: x[0], reverse=True)

        elites = [ir for _, ir in scored[: int(len(scored) * elite_ratio)]]

        best_ir = scored[0][1]
        paths = semantic_paths(best_ir)
        
        log_message = (
            f"Gen {gen}: best fitness = {scored[0][0]:.2f}, "
            f"nodes = {best_ir.num_nodes}, "
            f"edges = {best_ir.edge_index.size(1)}, "
            f"semantic_paths = {len(paths)}"
        )
        log_write(log_message)

        # Reproduce
        new_population = elites.copy()
        while len(new_population) < len(population):
            parent = random.choice(elites)
            child = semantic_mutate(parent, feature_dim)
            new_population.append(child)

        population = new_population

    return population


# =========================
# Initialization
# =========================

def initial_graph(feature_dim: int) -> GraphIR:
    """
    Create an initial graph with semantic meaning:
    INPUT -> COMPUTE -> OUTPUT
    """
    node_features = torch.randn(3, feature_dim)
    node_types = torch.tensor([
        NodeType.INPUT,
        NodeType.COMPUTE,
        NodeType.OUTPUT
    ])

    edge_index = torch.tensor([
        [0, 1],
        [1, 2]
    ], dtype=torch.long).t()
    
    # Assign random operation to COMPUTE node (node_id=1)
    node_ops = {1: random.choice(OP_NAMES)}

    return GraphIR(node_features, node_types, edge_index, node_ops=node_ops)


# =========================
# Entry point
# =========================

def main():
    # Setup logging
    log_path = setup_logging()
    log_write(f"Log file: {log_path}")
    log_write(f"Random seed: 0")
    log_write(f"Torch seed: 0")
    
    try:
        random.seed(0)
        torch.manual_seed(0)

        feature_dim = 8
        population_size = 10

        log_write(f"\n=== Configuration ===")
        log_write(f"Feature dimension: {feature_dim}")
        log_write(f"Population size: {population_size}")
        log_write(f"Generations: 10")

        # Initialize population
        log_write(f"\n=== Initialization ===")
        population = [
            initial_graph(feature_dim)
            for _ in range(population_size)
        ]
        log_write(f"Initialized population of {len(population)} graphs")

        # Run GA
        log_write(f"\n=== Evolution ===")
        evolved = evolve(
            population,
            generations=10,
            feature_dim=feature_dim
        )

        # Select best individual by fitness
        best_ir = max(evolved, key=fitness)
        best_fitness = fitness(best_ir)

        log_write(f"\n=== Best Individual ===")
        log_write(f"Best fitness: {best_fitness:.4f}")
        log_write(f"Number of nodes: {best_ir.num_nodes}")
        log_write(f"Number of edges: {best_ir.edge_index.size(1)}")
        log_write(f"COMPUTE node operations: {best_ir.node_ops}")
        
        paths = semantic_paths(best_ir)
        log_write(f"Semantic paths: {len(paths)}")
        if paths:
            log_write(f"Sample paths (first 3):")
            for i, path in enumerate(paths[:3]):
                log_write(f"  Path {i+1}: {path}")

        # Visualize best graph with fitness and path information
        visualize_graph(
            best_ir,
            title="Best Graph (Semantic Evolution)",
            fitness_value=best_fitness,
            show_paths=True
        )

        # Convert best individual to PyG Data
        pyg_data = ir_to_pyg(best_ir)

        log_write(f"\n=== PyG Data ===")
        log_write(str(pyg_data), also_print=False)
        
        # Execute the best graph
        log_write(f"\n=== Graph Execution ===")
        log_write(f"COMPUTE node operations: {best_ir.node_ops}")
        
        test_inputs = [1.0, 3.0, 5.0, 10.0]
        log_write(f"\nExecution results:")
        execution_results = []
        for inp in test_inputs:
            try:
                out = execute_graph(best_ir, inp)
                result_msg = f"  Input: {inp:5.1f} -> Output: {out:10.2f}"
                log_write(result_msg)
                execution_results.append((inp, out, None))
            except Exception as e:
                result_msg = f"  Input: {inp:5.1f} -> Error: {e}"
                log_write(result_msg)
                execution_results.append((inp, None, str(e)))
        
        # Compile to Python function
        log_write(f"\n=== AST Compilation ===")
        try:
            compiled_fn = compile_graph(best_ir)
            log_write("✓ Graph compiled to Python function")
            
            # Verify equivalence
            log_write(f"\nVerification (execute_graph vs compiled function):")
            verification_results = []
            for inp in test_inputs:
                try:
                    out1 = execute_graph(best_ir, inp)
                    out2 = compiled_fn(inp)
                    match = abs(out1 - out2) < 1e-10
                    result_msg = (
                        f"  Input: {inp:5.1f} -> execute_graph={out1:10.2f}, "
                        f"compiled_fn={out2:10.2f}, match={match}"
                    )
                    log_write(result_msg)
                    verification_results.append((inp, out1, out2, match))
                except Exception as e:
                    result_msg = f"  Input: {inp:5.1f} -> Error: {e}"
                    log_write(result_msg)
                    verification_results.append((inp, None, None, False))
            
            # Try to generate source code (Python 3.9+)
            try:
                source = ast_to_source(graph_to_ast(best_ir))
                log_write(f"\n✓ Generated Python source code ({len(source)} chars):")
                log_write("=" * 50)
                log_write(source, also_print=False)
                log_write("=" * 50)
            except NotImplementedError as e:
                log_write(f"\nNote: Source code generation not available ({e})")
        except Exception as e:
            log_write(f"✗ AST compilation failed: {e}")
        
        log_write(f"\n=== Summary ===")
        log_write(f"Best fitness achieved: {best_fitness:.4f}")
        log_write(f"Best graph structure: {best_ir.num_nodes} nodes, {best_ir.edge_index.size(1)} edges")
        log_write(f"Total semantic paths: {len(paths)}")
        log_write(f"Successful executions: {sum(1 for _, out, err in execution_results if err is None)}/{len(execution_results)}")
        
    finally:
        # Always close logging
        close_logging()
        print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()