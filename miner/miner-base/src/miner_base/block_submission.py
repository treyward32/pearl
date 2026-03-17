from miner_utils import get_logger
from pearl_gateway.comm.dataclasses import OpenedBlockInfo
from pearl_mining import MatrixMerkleProof, PlainProof

from .commitment_hash import CommitmentHasher
from .matrix_merkle_tree import MatrixMerkleTree

_LOGGER = get_logger(__name__)


def create_proof(
    opened_block_info: OpenedBlockInfo,
    incomplete_header_bytes: bytes,
) -> PlainProof:
    """Create a PlainProof from OpenedBlockInfo using non-noised A and B matrices."""
    _LOGGER.debug("Creating proof")
    A = opened_block_info.A
    B_t = opened_block_info.B_t
    mining_config = opened_block_info.get_mining_config()

    hash_key = CommitmentHasher.get_key(incomplete_header_bytes, mining_config)
    A_merkle_tree = MatrixMerkleTree(A, hash_key)
    B_merkle_tree = MatrixMerkleTree(B_t, hash_key)
    _LOGGER.debug("Generated merkle trees")

    a_merkle_proof = MatrixMerkleProof(
        proof=A_merkle_tree.get_multileaf_proof(
            A_merkle_tree.leaf_indices_from_rows(opened_block_info.A_row_indices)
        ),
        row_indices=opened_block_info.A_row_indices,
    )
    b_merkle_proof = MatrixMerkleProof(
        proof=B_merkle_tree.get_multileaf_proof(
            B_merkle_tree.leaf_indices_from_rows(opened_block_info.B_column_indices)
        ),
        row_indices=opened_block_info.B_column_indices,
    )

    m, k = A.shape
    n, k2 = B_t.shape
    assert k == k2, f"Common dimension mismatch: {k} != {k2}"

    return PlainProof(
        m=m,
        n=n,
        k=k,
        noise_rank=opened_block_info.noise_rank,
        a_merkle_proof=a_merkle_proof,
        bt_merkle_proof=b_merkle_proof,
    )
