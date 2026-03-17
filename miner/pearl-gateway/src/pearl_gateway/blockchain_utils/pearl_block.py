from bitcoinutils.transactions import Transaction
from bitcoinutils.utils import encode_varint, get_transaction_length, parse_compact_size
from pearl_gateway.blockchain_utils.pearl_header import PearlHeader
from pearl_gateway.blockchain_utils.zk_certificate import ZKCertificate


class PearlBlock:
    def __init__(self, header: PearlHeader, txns: list[Transaction], zk_certificate: ZKCertificate):
        self.header = header

        if header.proof_commitment is None:
            header.proof_commitment = zk_certificate.get_proof_commitment()
        elif header.proof_commitment != zk_certificate.get_proof_commitment():
            raise ValueError("Proof commitment mismatch")

        self.txns = txns
        self.zk_certificate = zk_certificate

    @staticmethod
    def _deserialize_transactions(data: bytes) -> list[Transaction]:
        txn_count, txn_offset = parse_compact_size(data)
        transactions = []
        for _ in range(txn_count):
            txn_len = get_transaction_length(data[txn_offset:])
            txn_bytes = data[txn_offset : txn_offset + txn_len]
            transactions.append(Transaction.from_raw(txn_bytes.hex()))
            txn_offset += txn_len
        return transactions

    @classmethod
    def deserialize(cls, data: bytes) -> "PearlBlock":
        """
        The structure is ZK_CERTIFICATE|PEARL_HEADER|TRANSACTIONS|
        """
        zk_certificate = ZKCertificate.deserialize(data)
        # zk certificate size is not constant, so we need to get it from the serialized data
        zk_certificate_size = zk_certificate.get_serialized_size()

        pearl_header_offset = zk_certificate_size
        pearl_header_size = PearlHeader.get_serialized_header_size()
        pearl_header_bytes = data[pearl_header_offset : pearl_header_offset + pearl_header_size]
        pearl_header = PearlHeader.deserialize(pearl_header_bytes)

        txn_offset = pearl_header_offset + pearl_header_size
        transactions = cls._deserialize_transactions(data[txn_offset:])
        return cls(pearl_header, transactions, zk_certificate)

    def serialize(self) -> bytes:
        """
        Format: ZK_CERTIFICATE|BLOCK_HEADER|TX_COUNT (varint)|TRANSACTIONS
        """
        zk_certificate_bytes = self.zk_certificate.serialize()
        header_bytes = self.header.serialize()
        tx_count_bytes = encode_varint(len(self.txns))
        transactions_bytes = b"".join(tx.to_bytes(tx.has_segwit) for tx in self.txns)
        return zk_certificate_bytes + header_bytes + tx_count_bytes + transactions_bytes
