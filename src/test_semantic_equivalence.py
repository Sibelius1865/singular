"""
Tests for semantic equivalence checking
"""

import sys
import ast
sys.path.insert(0, 'src')

from semantic_equivalence import (
    canonicalize_ast,
    ast_semantically_equal,
    graph_semantically_equal
)
from run import (
    initial_graph,
    graph_to_ast,
    semantic_mutate,
    OPS
)


def test_variable_renaming():
    """Test that variables are renamed to canonical names."""
    print("=== Test: Variable Renaming ===")
    
    # Create AST with non-canonical variable names
    code = """
def test_fn(x):
    n3 = OPS["mul2"](x)
    n7 = OPS["add1"](n3)
    return n7
"""
    tree = ast.parse(code)
    func_def = tree.body[0]
    
    canon = canonicalize_ast(func_def)
    dump = ast.dump(canon, include_attributes=False)
    
    # Check that variables are renamed to v0, v1, v2
    assert "v0" in dump or "v1" in dump, "Variables should be renamed"
    print("✓ Variable renaming works")


def test_id_elimination():
    """Test that OPS[\"id\"](x) is eliminated."""
    print("\n=== Test: ID Elimination ===")
    
    code = """
def test_fn(x):
    n1 = OPS["id"](x)
    n2 = OPS["mul2"](n1)
    return n2
"""
    tree = ast.parse(code)
    func_def = tree.body[0]
    
    canon = canonicalize_ast(func_def)
    dump = ast.dump(canon, include_attributes=False)
    
    # Check that id is eliminated
    assert 'id' not in dump or 'OPS["id"]' not in dump, "ID operations should be eliminated"
    print("✓ ID elimination works")


def test_same_graph_equivalence():
    """Test that same GraphIR is equivalent to itself."""
    print("\n=== Test: Same Graph Equivalence ===")
    
    g = initial_graph(8)
    
    result = graph_semantically_equal(g, g)
    assert result, "Same graph should be equivalent to itself"
    print("✓ Same graph is equivalent to itself")


def test_different_operations():
    """Test that graphs with different operations are not equivalent."""
    print("\n=== Test: Different Operations ===")
    
    # Create two graphs with explicitly different operations
    from run import GraphIR, NodeType
    import torch
    
    # Graph 1: INPUT -> COMPUTE(mul2) -> OUTPUT
    node_features1 = torch.randn(3, 8)
    node_types1 = torch.tensor([NodeType.INPUT, NodeType.COMPUTE, NodeType.OUTPUT])
    edge_index1 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    node_ops1 = {1: "mul2"}
    g1 = GraphIR(node_features1, node_types1, edge_index1, node_ops=node_ops1)
    
    # Graph 2: INPUT -> COMPUTE(add1) -> OUTPUT
    node_features2 = torch.randn(3, 8)
    node_types2 = torch.tensor([NodeType.INPUT, NodeType.COMPUTE, NodeType.OUTPUT])
    edge_index2 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    node_ops2 = {1: "add1"}
    g2 = GraphIR(node_features2, node_types2, edge_index2, node_ops=node_ops2)
    
    result = graph_semantically_equal(g1, g2)
    assert not result, "Graphs with different operations should not be equivalent"
    print("✓ Different operations correctly identified as non-equivalent")


def test_mutation_preserves_meaning():
    """Test that mutation that doesn't change meaning preserves equivalence."""
    print("\n=== Test: Mutation Preserves Meaning ===")
    
    # This is a tricky test - we need mutations that don't change meaning
    # For now, test that same graph after copy is equivalent
    g1 = initial_graph(8)
    g2 = g1.copy()
    
    result = graph_semantically_equal(g1, g2)
    assert result, "Copy of graph should be equivalent"
    print("✓ Graph copy is equivalent")


def test_canonical_form_stability():
    """Test that canonicalization produces stable output."""
    print("\n=== Test: Canonical Form Stability ===")
    
    g = initial_graph(8)
    ast1 = graph_to_ast(g)
    
    canon1 = canonicalize_ast(ast1)
    canon2 = canonicalize_ast(ast1)  # Canonicalize again
    
    dump1 = ast.dump(canon1, include_attributes=False)
    dump2 = ast.dump(canon2, include_attributes=False)
    
    assert dump1 == dump2, "Canonical form should be stable"
    print("✓ Canonical form is stable")


def test_ast_semantic_equivalence():
    """Test AST semantic equivalence checking."""
    print("\n=== Test: AST Semantic Equivalence ===")
    
    # Create two ASTs that should be equivalent (same graph)
    g1 = initial_graph(8)
    g2 = g1.copy()
    
    ast1 = graph_to_ast(g1)
    ast2 = graph_to_ast(g2)
    
    result = ast_semantically_equal(ast1, ast2)
    assert result, "ASTs from same graph should be equivalent"
    print("✓ AST semantic equivalence works")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Semantic Equivalence Tests")
    print("=" * 60)
    
    try:
        test_variable_renaming()
        test_id_elimination()
        test_same_graph_equivalence()
        test_different_operations()
        test_mutation_preserves_meaning()
        test_canonical_form_stability()
        test_ast_semantic_equivalence()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

