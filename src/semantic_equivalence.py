"""
AST Normalization and Semantic Equivalence Checking

This module provides functionality to:
1. Normalize ASTs to canonical forms
2. Check semantic equivalence between ASTs and GraphIRs
"""

import ast
import copy
from typing import Dict, List, Set, Optional
from collections import defaultdict, deque


# =========================
# AST Normalization
# =========================

class VariableRenamer(ast.NodeTransformer):
    """
    Rename variables to canonical names (v0, v1, v2, ...)
    based on their order of first assignment.
    """
    
    def __init__(self):
        self.var_map: Dict[str, str] = {}
        self.counter = 0
        self.seen_vars: Set[str] = set()
        self.assignment_order: List[str] = []
    
    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Rename variable references."""
        if isinstance(node.ctx, ast.Store):
            # Assignment target
            if node.id not in self.var_map:
                new_name = f"v{self.counter}"
                self.var_map[node.id] = new_name
                self.assignment_order.append(node.id)
                self.counter += 1
            return ast.Name(id=self.var_map[node.id], ctx=node.ctx)
        elif isinstance(node.ctx, ast.Load):
            # Variable reference
            if node.id in self.var_map:
                return ast.Name(id=self.var_map[node.id], ctx=node.ctx)
            # Function parameter (e.g., 'x') - don't rename
            return node
        return node


class IdEliminator(ast.NodeTransformer):
    """
    Eliminate OPS["id"](x) -> x
    """
    
    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Replace OPS["id"](x) with x."""
        # Check if this is OPS["id"](...)
        if isinstance(node.func, ast.Subscript):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "OPS":
                if isinstance(node.func.slice, ast.Constant):
                    if node.func.slice.value == "id":
                        # OPS["id"](x) -> x
                        if len(node.args) == 1:
                            return self.visit(node.args[0])
        
        # Recursively process children
        return self.generic_visit(node)


class DeadCodeEliminator:
    """
    Remove assignments to variables that are never used.
    Uses two-pass approach: first collect usage, then remove unused.
    """
    
    def __init__(self):
        self.used_vars: Set[str] = set()
    
    def collect_usage(self, node: ast.AST):
        """First pass: collect all variable usages."""
        class UsageCollector(ast.NodeVisitor):
            def __init__(self, used_vars):
                self.used_vars = used_vars
            
            def visit_Name(self, node: ast.Name):
                if isinstance(node.ctx, ast.Load):
                    self.used_vars.add(node.id)
                self.generic_visit(node)
        
        collector = UsageCollector(self.used_vars)
        collector.visit(node)
    
    def eliminate(self, node: ast.AST) -> ast.AST:
        """Second pass: remove unused assignments."""
        class Eliminator(ast.NodeTransformer):
            def __init__(self, used_vars):
                self.used_vars = used_vars
            
            def visit_Assign(self, node: ast.Assign) -> Optional[ast.Assign]:
                # Check if any target is used
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id in self.used_vars:
                            # At least one target is used, keep the assignment
                            return self.generic_visit(node)
                # No targets are used, remove this assignment
                return None
        
        eliminator = Eliminator(self.used_vars)
        return eliminator.visit(node)


class TopologicalSorter:
    """
    Reorder statements to ensure topological order based on dependencies.
    """
    
    def __init__(self):
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.assignments: Dict[str, ast.Assign] = {}
        self.return_stmt: Optional[ast.Return] = None
    
    def analyze(self, body: List[ast.stmt]):
        """Analyze dependencies between statements."""
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                # Get assigned variable name
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    var_name = stmt.targets[0].id
                    self.assignments[var_name] = stmt
                    
                    # Find dependencies (variables used in the value)
                    deps = self._extract_dependencies(stmt.value)
                    self.dependencies[var_name] = deps
            elif isinstance(stmt, ast.Return):
                self.return_stmt = stmt
    
    def _extract_dependencies(self, node: ast.AST) -> Set[str]:
        """Extract variable names used in an expression."""
        deps = set()
        
        class DependencyCollector(ast.NodeVisitor):
            def visit_Name(self, node: ast.Name):
                if isinstance(node.ctx, ast.Load):
                    deps.add(node.id)
        
        collector = DependencyCollector()
        collector.visit(node)
        return deps
    
    def sort(self) -> List[ast.stmt]:
        """Sort statements in topological order."""
        sorted_stmts = []
        remaining = set(self.assignments.keys())
        processed = set()
        
        # Process in topological order
        while remaining:
            # Find variables with no unprocessed dependencies
            ready = [
                var for var in remaining
                if self.dependencies[var].issubset(processed)
            ]
            
            if not ready:
                # Circular dependency or missing dependency - use original order
                ready = list(remaining)
            
            # Process ready variables in sorted order for stability
            for var in sorted(ready):
                sorted_stmts.append(self.assignments[var])
                processed.add(var)
                remaining.remove(var)
        
        # Add return statement at the end
        if self.return_stmt:
            sorted_stmts.append(self.return_stmt)
        
        return sorted_stmts


def canonicalize_ast(fn_ast: ast.AST) -> ast.AST:
    """
    Convert a function AST to canonical form.
    
    Steps:
    1. Rename variables to canonical names (v0, v1, v2, ...)
    2. Eliminate OPS["id"](x) -> x
    3. Remove dead code
    4. Sort statements in topological order
    
    Args:
        fn_ast: Function AST (ast.FunctionDef or ast.Module containing one)
    
    Returns:
        Canonicalized AST (ast.FunctionDef)
    """
    # Extract function definition
    if isinstance(fn_ast, ast.Module):
        if not fn_ast.body or not isinstance(fn_ast.body[0], ast.FunctionDef):
            raise ValueError("Module does not contain a function definition")
        func_def = fn_ast.body[0]
    elif isinstance(fn_ast, ast.FunctionDef):
        func_def = fn_ast
    else:
        raise ValueError(f"Expected FunctionDef or Module, got {type(fn_ast)}")
    
    # Create a deep copy to avoid modifying original
    func_def = ast.fix_missing_locations(ast.copy_location(
        copy.deepcopy(func_def), func_def
    ))
    
    # Step 1: Rename variables
    renamer = VariableRenamer()
    func_def = renamer.visit(func_def)
    
    # Step 2: Eliminate id operations
    eliminator = IdEliminator()
    func_def = eliminator.visit(func_def)
    
    # Step 3: Remove dead code
    dead_eliminator = DeadCodeEliminator()
    # First pass: collect usage
    dead_eliminator.collect_usage(func_def)
    # Second pass: remove unused assignments
    func_def = dead_eliminator.eliminate(func_def)
    
    # Step 4: Sort statements topologically
    sorter = TopologicalSorter()
    sorter.analyze(func_def.body)
    func_def.body = sorter.sort()
    
    # Fix missing locations
    ast.fix_missing_locations(func_def)
    
    return func_def


# =========================
# Semantic Equivalence
# =========================

def ast_semantically_equal(ast1: ast.AST, ast2: ast.AST) -> bool:
    """
    Check if two ASTs are semantically equivalent.
    
    Strategy:
    1. Normalize both ASTs to canonical form
    2. Compare normalized ASTs using ast.dump
    3. If different, fallback to execution-based comparison
    
    Args:
        ast1: First AST
        ast2: Second AST
    
    Returns:
        True if semantically equivalent, False otherwise
    """
    # Step 1: Normalize both ASTs
    try:
        canon1 = canonicalize_ast(ast1)
        canon2 = canonicalize_ast(ast2)
    except Exception as e:
        # If normalization fails, fallback to execution test
        return _ast_equal_by_execution(ast1, ast2)
    
    # Step 2: Compare normalized ASTs
    dump1 = ast.dump(canon1, include_attributes=False)
    dump2 = ast.dump(canon2, include_attributes=False)
    
    if dump1 == dump2:
        return True
    
    # Step 3: Fallback to execution-based comparison
    return _ast_equal_by_execution(ast1, ast2)


def _ast_equal_by_execution(ast1: ast.AST, ast2: ast.AST) -> bool:
    """
    Compare ASTs by executing them on a set of test inputs.
    
    Args:
        ast1: First AST
        ast2: Second AST
    
    Returns:
        True if both produce same outputs for all test inputs
    """
    # Import OPS from run module
    try:
        from .run import OPS
    except ImportError:
        from run import OPS
    
    # Test inputs - use more diverse set
    test_inputs = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0, 100.0]
    
    try:
        # Compile and execute ast1
        if isinstance(ast1, ast.Module):
            module1 = ast1
        elif isinstance(ast1, ast.FunctionDef):
            module1 = ast.Module(body=[ast1], type_ignores=[])
        else:
            return False
        
        ast.fix_missing_locations(module1)
        code1 = compile(module1, "<ast1>", "exec")
        
        namespace1 = {"OPS": OPS}
        exec(code1, namespace1)
        fn1 = namespace1.get("generated_fn")
        if fn1 is None:
            return False
        
        # Compile and execute ast2
        if isinstance(ast2, ast.Module):
            module2 = ast2
        elif isinstance(ast2, ast.FunctionDef):
            module2 = ast.Module(body=[ast2], type_ignores=[])
        else:
            return False
        
        ast.fix_missing_locations(module2)
        code2 = compile(module2, "<ast2>", "exec")
        
        namespace2 = {"OPS": OPS}
        exec(code2, namespace2)
        fn2 = namespace2.get("generated_fn")
        if fn2 is None:
            return False
        
        # Compare outputs
        for x in test_inputs:
            try:
                y1 = fn1(x)
                y2 = fn2(x)
                # Use relative tolerance for floating point comparison
                if abs(y1 - y2) > max(1e-10, 1e-10 * max(abs(y1), abs(y2))):
                    return False
            except Exception:
                return False
        
        return True
    
    except Exception:
        return False


def graph_semantically_equal(g1, g2) -> bool:
    """
    Check if two GraphIRs are semantically equivalent.
    
    Args:
        g1: First GraphIR
        g2: Second GraphIR
    
    Returns:
        True if semantically equivalent, False otherwise
    """
    try:
        from .run import graph_to_ast
    except ImportError:
        from run import graph_to_ast
    
    try:
        ast1 = graph_to_ast(g1)
        ast2 = graph_to_ast(g2)
        return ast_semantically_equal(ast1, ast2)
    except Exception:
        return False

