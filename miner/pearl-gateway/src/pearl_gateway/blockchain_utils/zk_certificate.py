from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from pearl_mining import PUBLICDATA_SIZE, ZKProof

from .blockchain_utils import double_sha256
from .pearl_header import PearlHeader


@dataclass
class ZKCertificate:
    header_hash: bytes
    proof: ZKProof

    # Fixed header dtype (proof_data is variable length, handled separately)
    DTYPE_HEADER: ClassVar[np.dtype] = np.dtype(
        [
            ("version", "<u4"),
            ("header_hash", "V32"),
            ("public_data", f"V{PUBLICDATA_SIZE}"),
            ("proof_data_len", "<u4"),
        ]
    )

    # Certificate version for ZK certificates
    ZK_CERTIFICATE_VERSION: ClassVar[int] = 1
    ZK_MAX_PROOF_DATA_SIZE: ClassVar[int] = 60000

    def __post_init__(self):
        if len(self.proof.public_data) != PUBLICDATA_SIZE:
            raise ValueError(
                f"public_data must be exactly {PUBLICDATA_SIZE} bytes, got {len(self.proof.public_data)}"
            )
        if len(self.proof.proof_data) > self.ZK_MAX_PROOF_DATA_SIZE:
            raise ValueError(
                f"Proof data is too large: {len(self.proof.proof_data)} bytes (max {self.ZK_MAX_PROOF_DATA_SIZE} bytes)"
            )

    def serialize(self) -> bytes:
        arr = np.array(
            [
                (
                    self.ZK_CERTIFICATE_VERSION,
                    self.header_hash,
                    self.proof.public_data,
                    len(self.proof.proof_data),
                )
            ],
            dtype=self.DTYPE_HEADER,
        )
        return arr.tobytes() + bytes(self.proof.proof_data)

    def get_serialized_size(self) -> int:
        return self.DTYPE_HEADER.itemsize + len(self.proof.proof_data)

    @classmethod
    def deserialize(cls, data: bytes) -> "ZKCertificate":
        arr = np.frombuffer(data, dtype=cls.DTYPE_HEADER, count=1)[0]
        proof_data_len = int(arr["proof_data_len"])
        proof_data = data[cls.DTYPE_HEADER.itemsize : cls.DTYPE_HEADER.itemsize + proof_data_len]
        return cls(
            header_hash=bytes(arr["header_hash"]),
            proof=ZKProof(bytes(arr["public_data"]), bytes(proof_data)),
        )

    @classmethod
    def from_pearl_header(
        cls,
        header: PearlHeader,
        proof: ZKProof,
    ) -> "ZKCertificate":
        commitment = cls._get_proof_commitment(proof.public_data)
        if header.proof_commitment is None:
            header.proof_commitment = commitment
        elif header.proof_commitment != commitment:
            raise ValueError("Proof commitment mismatch")

        return cls(
            header_hash=double_sha256(header.serialize()),
            proof=proof,
        )

    @staticmethod
    def _get_proof_commitment(public_data: bytes) -> bytes:
        version_bytes = ZKCertificate.ZK_CERTIFICATE_VERSION.to_bytes(4, "little")
        return double_sha256(version_bytes + public_data)

    def get_proof_commitment(self) -> bytes:
        return self._get_proof_commitment(self.proof.public_data)
