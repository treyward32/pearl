import pytest
import torch
from blake3 import blake3
from miner_base.matrix_merkle_tree import MatrixMerkleTree
from pearl_gemm import commitment_hash_from_merkle_roots, get_required_scratchpad_bytes, tensor_hash


def hash_matrix(matrix: torch.Tensor, key: bytes) -> torch.Tensor:
    """Reference implementation for tensor hash."""
    hash_bytes = MatrixMerkleTree.tensor_hash(matrix, key)
    return torch.frombuffer(hash_bytes, dtype=torch.uint8).to("cuda")


@pytest.fixture(autouse=True)
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TestTensorHash:
    """Test tensor hash functionality on different matrix sizes."""

    @pytest.mark.parametrize(
        "shape",
        [
            (8192, 8192),
            (1337, 8192),
            (4096, 2048),
            (2048, 4096),
            (512, 512),
            (777, 1024),
            (2048, 3072),
            (
                2000,
                2048,
            ),
            (7952, 1024),
            (7984, 1024),
            (8192, 28672),
            (28672, 8192),
            (57344, 8192),
            (8192, 57344),
            (16384, 57344),
            (57344, 16384),
            (1245, 5136),
            (12451, 23141),
            (12345, 22141),
        ],
    )
    def test_tensor_hash_shapes(self, shape):
        """Test that CUDA implementation matches Python reference for various shapes."""
        # Create random tensor with the given shape
        matrix = torch.randint(0, 255, shape, dtype=torch.uint8, device="cuda")

        # Dynamically allocate scratchpad based on matrix size
        scratchpad_size = get_required_scratchpad_bytes(matrix.numel())
        scratchpad = torch.empty(scratchpad_size, dtype=torch.uint8, device="cuda")

        # Create random key (32 bytes for Blake3)
        cuda_result = torch.empty(32, dtype=torch.uint8, device="cuda")
        key_tensor = torch.randint(0, 255, (32,), dtype=torch.uint8, device="cuda")
        key_bytes = key_tensor.cpu().numpy().tobytes()

        # Compute hash using CUDA implementation
        tensor_hash(matrix, key_tensor, cuda_result, scratchpad)
        torch.cuda.synchronize()

        # Compute hash using Python reference
        python_result = hash_matrix(matrix.cpu(), key_bytes)

        # Compare results
        assert torch.equal(cuda_result, python_result), (
            f"Hash mismatch for shape {shape}: CUDA result doesn't match Python reference"
        )

    def test_commitment_hash_from_merkle_roots(self):
        """Test that commitment hash from merkle roots matches Python reference."""

        A_merkle_root = torch.randint(0, 255, (32,), dtype=torch.uint8, device="cuda")
        B_merkle_root = torch.randint(0, 255, (32,), dtype=torch.uint8, device="cuda")
        key = torch.randint(0, 255, (32,), dtype=torch.uint8, device="cuda")
        cuda_result_A = torch.empty(32, dtype=torch.uint8, device="cuda")
        cuda_result_B = torch.empty(32, dtype=torch.uint8, device="cuda")
        commitment_hash_from_merkle_roots(
            A_merkle_root, B_merkle_root, key, cuda_result_A, cuda_result_B
        )
        torch.cuda.synchronize()

        python_result_B = blake3(
            key.cpu().numpy().tobytes() + B_merkle_root.cpu().numpy().tobytes()
        ).digest()
        python_result_A = blake3(python_result_B + A_merkle_root.cpu().numpy().tobytes()).digest()

        assert cuda_result_A.cpu().numpy().tobytes() == python_result_A, (
            "Commitment hash from merkle roots mismatch: CUDA result doesn't match Python reference"
        )
        assert cuda_result_B.cpu().numpy().tobytes() == python_result_B, (
            "Commitment hash from merkle roots mismatch: CUDA result doesn't match Python reference"
        )

    @pytest.mark.parametrize(
        "shape",
        [
            (8192, 8192),  # Large square matrix (8k x 8k)
            (1337, 8192),  # Irregular dimensions
            (4096, 2048),  # Rectangular matrix
            (2048, 4096),  # Rectangular matrix (transposed)
            # (512, 512) is excluded as it doesn't work with 512 threads
            (777, 1024),  # Another irregular shape
            (2048, 3072),  # Non-power-of-2 dimensions
            (
                2000,
                2048,
            ),  # Initial test case we encountered the bug with (31 complete blocks and remainder block with R=32)
            (7952, 1024),  # 62 complete blocks and remainder blocks with R=64
            (7984, 1024),  # 62 complete blocks and remainder blocks with R=48=32+16
            (8192, 28672),
            (28672, 8192),
            (57344, 8192),
            (8192, 57344),
            (16384, 57344),
            (57344, 16384),
            (1245, 5136),
            (12451, 23141),
            (12345, 22141),
        ],
    )
    def test_tensor_hash_512_threads(self, shape):
        """Test tensor hash with 512 threads per block."""

        # Create random tensor with the given shape
        matrix = torch.randint(0, 255, shape, dtype=torch.uint8, device="cuda")

        # Dynamically allocate scratchpad based on matrix size
        scratchpad_size = get_required_scratchpad_bytes(matrix.numel())
        scratchpad = torch.empty(scratchpad_size, dtype=torch.uint8, device="cuda")

        # Create random key (32 bytes for Blake3)
        cuda_result = torch.empty(32, dtype=torch.uint8, device="cuda")
        key_tensor = torch.randint(0, 255, (32,), dtype=torch.uint8, device="cuda")
        key_bytes = key_tensor.cpu().numpy().tobytes()

        # Compute hash using CUDA implementation with 512 threads per block
        tensor_hash(matrix, key_tensor, cuda_result, scratchpad, threads_per_block=512)
        torch.cuda.synchronize()

        # Compute hash using Python reference
        python_result = hash_matrix(matrix, key_bytes)

        # Compare results
        assert torch.equal(cuda_result, python_result), (
            f"Hash mismatch for shape {shape} with 512 threads: CUDA result doesn't match Python reference"
        )

    def test_tensor_hash_deterministic(self):
        """Test that tensor hash produces deterministic results."""
        shape = (8192, 8192)
        data = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
        key = torch.zeros(32, dtype=torch.uint8, device="cuda")

        # Dynamically allocate scratchpad based on matrix size
        scratchpad_size = get_required_scratchpad_bytes(data.numel())
        scratchpad = torch.empty(scratchpad_size, dtype=torch.uint8, device="cuda")

        base_out = torch.zeros(32, dtype=torch.uint8, device="cuda")
        tensor_hash(data, key, base_out, scratchpad)

        for _ in range(100000):
            new_out = torch.zeros(32, dtype=torch.uint8, device="cuda")
            tensor_hash(data, key, new_out, scratchpad)

            assert torch.equal(base_out, new_out), "Hash mismatch"
