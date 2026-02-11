#!/usr/bin/env python3
"""Verify embedding module structure without requiring torch/sentence-transformers."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_module_structure() -> None:
    """Test embedding module structure and imports."""
    print("=== Verifying Embedding Module Structure ===\n")

    # Test 1: Models
    print("Test 1: Verifying models...")
    from inference.embedding.models import EmbeddingBatch, EmbeddingResult

    # Create test result
    result = EmbeddingResult(chunk_id="test_001", embedding=[0.1, 0.2, 0.3])
    assert result.chunk_id == "test_001"
    assert len(result.embedding) == 3
    print("  ✓ EmbeddingResult works")

    # Create test batch
    batch = EmbeddingBatch(results=[result], model_name="test-model", dimension=3)
    assert len(batch.results) == 1
    assert batch.model_name == "test-model"
    assert batch.dimension == 3
    print("  ✓ EmbeddingBatch works")

    # Test dimension validation
    try:
        EmbeddingBatch(
            results=[EmbeddingResult(chunk_id="test", embedding=[0.1, 0.2])],
            model_name="test",
            dimension=3,  # Mismatch!
        )
        print("  ✗ Dimension validation failed")
        sys.exit(1)
    except ValueError as e:
        assert "dimension mismatch" in str(e).lower()
        print("  ✓ Dimension validation works")

    # Test 2: Module exports (models only)
    print("\nTest 2: Verifying module exports...")
    from inference.embedding import EmbeddingBatch, EmbeddingResult

    print("  ✓ Model exports available")
    print("  ℹ BGEEmbedder and Embedder require torch/numpy (skip on macOS x86_64)")

    # Test 3: Check module __all__
    print("\nTest 3: Verifying __all__ exports...")
    import inference.embedding as embedding_module

    assert hasattr(embedding_module, "__all__")
    expected_exports = {"BGEEmbedder", "Embedder", "EmbeddingBatch", "EmbeddingResult"}
    actual_exports = set(embedding_module.__all__)
    assert actual_exports == expected_exports, (
        f"Expected {expected_exports}, got {actual_exports}"
    )
    print(f"  ✓ __all__ exports: {', '.join(sorted(actual_exports))}")

    # Test 4: Constants (read from source file)
    print("\nTest 4: Verifying constants...")
    from pathlib import Path

    bge_embedder_path = (
        Path(__file__).parent.parent
        / "src"
        / "inference"
        / "embedding"
        / "bge_embedder.py"
    )
    source = bge_embedder_path.read_text()

    assert 'MODEL_NAME = "BAAI/bge-m3"' in source
    assert "MAX_BATCH_SIZE = 64" in source
    assert "EMBEDDING_DIMENSION = 1024" in source
    print("  ✓ MODEL_NAME: BAAI/bge-m3")
    print("  ✓ MAX_BATCH_SIZE: 64")
    print("  ✓ EMBEDDING_DIMENSION: 1024")

    print("\n=== All Structure Tests Passed! ===")
    print("\nNote: Full functional tests require torch and sentence-transformers.")
    print("Run 'scripts/test_embedding.py' on Linux with GPU for full verification.")


if __name__ == "__main__":
    test_module_structure()
