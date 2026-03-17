import torch
from blake3 import blake3
from pearl_gateway.comm.dataclasses import CommitmentHash
from pearl_mining import MiningConfiguration

from .matrix_merkle_tree import MatrixMerkleTree


class CommitmentHasher:
    """
    A namespace for commitment hash functions.
    """

    @staticmethod
    def get_key(incomplete_header_bytes: bytes, mining_config: MiningConfiguration) -> bytes:
        return blake3(incomplete_header_bytes + mining_config.to_bytes()).digest()

    @classmethod
    def commitment_hash(
        cls,
        A: torch.Tensor,
        B: torch.Tensor,
        incomplete_header_bytes: bytes,
        mining_config: MiningConfiguration,
    ) -> CommitmentHash:
        key = cls.get_key(incomplete_header_bytes, mining_config)
        merkle_tree_A = MatrixMerkleTree(A, key)

        # We hash B.T because we would like to expose a column strip of B
        merkle_tree_B = MatrixMerkleTree(B.T, key)

        return cls.commitment_hash_from_merkle_roots(merkle_tree_A.root, merkle_tree_B.root, key)

    @staticmethod
    def get_commitment_B_key(key: bytes, B_merkle_root: bytes) -> bytes:
        return blake3(key + B_merkle_root).digest()

    @staticmethod
    def get_commitment_A_key(commitment_B: bytes, A_merkle_root: bytes) -> bytes:
        return blake3(commitment_B + A_merkle_root).digest()

    @classmethod
    def commitment_hash_from_merkle_roots(
        cls, A_merkle_root: bytes, B_merkle_root: bytes, key: bytes
    ) -> CommitmentHash:
        commitment_B = cls.get_commitment_B_key(key, B_merkle_root)
        commitment_A = cls.get_commitment_A_key(commitment_B, A_merkle_root)
        return CommitmentHash(commitment_A, commitment_B)
