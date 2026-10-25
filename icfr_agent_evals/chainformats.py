"""Wallet-address and transaction-hash encoders for the five chains in scope.

The generator draws random key material and encodes it here, so every address it
emits decodes with a valid checksum in its chain's real format, and every address
and hash matches casefile.ADDRESS_FORMATS and casefile.TX_HASH_FORMATS (the forms
the narrative sweep accepts, pre-registration section 6):

  - bitcoin: bech32 (BIP-173) P2WPKH, "bc1q" + 38 characters;
  - cardano: bech32 Shelley addresses with hrp "addr" (base "addr1q...", 103
    characters, and enterprise "addr1v...", 58 characters), CIP-19 headers;
  - ethereum (ETH, USDC and USDT here): "0x" + 40 lowercase hex. EIP-55 checksum
    casing would need Keccak-256, which hashlib does not provide (sha3_256 is the
    FIPS 202 variant, not Keccak), so addresses stay all lowercase;
  - solana: base58 (Bitcoin dictionary) of a 32-byte public key;
  - xrpl: classic addresses, base58check over version byte 0x00 and a 20-byte
    account id, with the XRP dictionary and a double-SHA-256 checksum.

Decoders are provided so tests can verify every checksum. No network access.
"""

from __future__ import annotations

import hashlib
import random

from .casefile import ADDRESS_FORMATS, BASE58_BITCOIN, BASE58_XRP, BECH32_CHARS, TX_HASH_FORMATS, matches_format

# --- bech32 (BIP-173) ----------------------------------------------------------------

_BECH32_GEN = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)


def _bech32_polymod(values: list[int]) -> int:
    chk = 1
    for v in values:
        top = chk >> 25
        chk = ((chk & 0x1FFFFFF) << 5) ^ v
        for i in range(5):
            if (top >> i) & 1:
                chk ^= _BECH32_GEN[i]
    return chk


def _hrp_expand(hrp: str) -> list[int]:
    return [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]


def convertbits(data: bytes | list[int], frombits: int, tobits: int, pad: bool = True) -> list[int]:
    """Regroup a sequence of frombits-wide values into tobits-wide values."""
    acc = 0
    bits = 0
    out: list[int] = []
    maxv = (1 << tobits) - 1
    for value in data:
        if value < 0 or value >> frombits:
            raise ValueError("value out of range for convertbits")
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            out.append((acc >> bits) & maxv)
    if pad:
        if bits:
            out.append((acc << (tobits - bits)) & maxv)
    elif bits >= frombits or ((acc << (tobits - bits)) & maxv):
        raise ValueError("invalid padding in convertbits")
    return out


def bech32_encode(hrp: str, data: list[int]) -> str:
    """bech32 string for an hrp and 5-bit data values (checksum appended)."""
    values = _hrp_expand(hrp) + list(data)
    polymod = _bech32_polymod(values + [0] * 6) ^ 1
    checksum = [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp + "1" + "".join(BECH32_CHARS[d] for d in list(data) + checksum)


def bech32_decode(text: str) -> tuple[str, list[int]]:
    """(hrp, data values without checksum). Raises ValueError on a bad checksum,
    mixed case or a character outside the bech32 set. No length cap, because
    Cardano addresses exceed BIP-173's 90-character limit."""
    if text.lower() != text and text.upper() != text:
        raise ValueError("mixed case")
    text = text.lower()
    pos = text.rfind("1")
    if pos < 1 or pos + 7 > len(text):
        raise ValueError("separator position")
    hrp, payload = text[:pos], text[pos + 1 :]
    if any(c not in BECH32_CHARS for c in payload):
        raise ValueError("character outside the bech32 set")
    data = [BECH32_CHARS.index(c) for c in payload]
    if _bech32_polymod(_hrp_expand(hrp) + data) != 1:
        raise ValueError("bad bech32 checksum")
    return hrp, data[:-6]


def p2wpkh_address(pubkey_hash: bytes, hrp: str = "bc") -> str:
    """Native segwit v0 address for a 20-byte key hash."""
    if len(pubkey_hash) != 20:
        raise ValueError("P2WPKH needs a 20-byte program")
    return bech32_encode(hrp, [0] + convertbits(pubkey_hash, 8, 5))


def decode_segwit(address: str, hrp: str = "bc") -> tuple[int, bytes]:
    """(witness version, program) of a segwit address."""
    got_hrp, data = bech32_decode(address)
    if got_hrp != hrp or not data:
        raise ValueError("wrong hrp")
    program = bytes(convertbits(data[1:], 5, 8, pad=False))
    if not 2 <= len(program) <= 40:
        raise ValueError("program length")
    if data[0] == 0 and len(program) not in (20, 32):
        raise ValueError("v0 program length")
    return data[0], program


# CIP-19 header bytes on mainnet (network id 1): type 0 base address (payment key
# hash, stake key hash) and type 6 enterprise address (payment key hash only).
CARDANO_BASE_HEADER = 0x01
CARDANO_ENTERPRISE_HEADER = 0x61


def cardano_address(payment_hash: bytes, stake_hash: bytes | None = None) -> str:
    """Shelley mainnet address: base when a stake key hash is given, else enterprise."""
    if len(payment_hash) != 28 or (stake_hash is not None and len(stake_hash) != 28):
        raise ValueError("Cardano key hashes are 28 bytes")
    if stake_hash is None:
        raw = bytes([CARDANO_ENTERPRISE_HEADER]) + payment_hash
    else:
        raw = bytes([CARDANO_BASE_HEADER]) + payment_hash + stake_hash
    return bech32_encode("addr", convertbits(raw, 8, 5))


def decode_cardano(address: str) -> bytes:
    """Raw address bytes (header first) of a mainnet Shelley address."""
    hrp, data = bech32_decode(address)
    if hrp != "addr":
        raise ValueError("wrong hrp")
    raw = bytes(convertbits(data, 5, 8, pad=False))
    if raw[0] == CARDANO_BASE_HEADER and len(raw) == 57:
        return raw
    if raw[0] == CARDANO_ENTERPRISE_HEADER and len(raw) == 29:
        return raw
    raise ValueError("unsupported header or length")


# --- base58 and base58check ----------------------------------------------------------

def base58_encode(raw: bytes, alphabet: str = BASE58_BITCOIN) -> str:
    n = int.from_bytes(raw, "big")
    out = []
    while n:
        n, rem = divmod(n, 58)
        out.append(alphabet[rem])
    leading = len(raw) - len(raw.lstrip(b"\0"))
    return alphabet[0] * leading + "".join(reversed(out))


def base58_decode(text: str, alphabet: str = BASE58_BITCOIN) -> bytes:
    n = 0
    for c in text:
        idx = alphabet.find(c)
        if idx < 0:
            raise ValueError(f"character {c!r} outside the base58 dictionary")
        n = n * 58 + idx
    body = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    leading = len(text) - len(text.lstrip(alphabet[0]))
    return b"\0" * leading + body


def _double_sha256(raw: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(raw).digest()).digest()


def base58check_encode(version: bytes, payload: bytes, alphabet: str) -> str:
    raw = version + payload
    return base58_encode(raw + _double_sha256(raw)[:4], alphabet)


def base58check_decode(text: str, alphabet: str) -> bytes:
    """version + payload. Raises ValueError on a bad checksum."""
    raw = base58_decode(text, alphabet)
    if len(raw) < 5:
        raise ValueError("too short")
    body, check = raw[:-4], raw[-4:]
    if _double_sha256(body)[:4] != check:
        raise ValueError("bad base58check checksum")
    return body


def xrp_classic_address(account_id: bytes) -> str:
    if len(account_id) != 20:
        raise ValueError("XRP account ids are 20 bytes")
    return base58check_encode(b"\x00", account_id, BASE58_XRP)


def decode_xrp(address: str) -> bytes:
    """The 20-byte account id of a classic XRP address."""
    body = base58check_decode(address, BASE58_XRP)
    if body[0] != 0 or len(body) != 21:
        raise ValueError("not a classic account address")
    return body[1:]


def solana_address(pubkey: bytes) -> str:
    if len(pubkey) != 32:
        raise ValueError("Solana public keys are 32 bytes")
    return base58_encode(pubkey, BASE58_BITCOIN)


def evm_address(account: bytes) -> str:
    if len(account) != 20:
        raise ValueError("EVM accounts are 20 bytes")
    return "0x" + account.hex()


# --- random draws --------------------------------------------------------------------

def random_address(chain: str, rng: random.Random, *, style: str = "default") -> str:
    """A checksum-valid address on `chain` from rng-drawn key material. For cardano,
    style "enterprise" gives an addr1v address, otherwise a base addr1q address."""
    while True:
        if chain == "bitcoin":
            addr = p2wpkh_address(rng.randbytes(20))
        elif chain == "ethereum":
            addr = evm_address(rng.randbytes(20))
        elif chain == "solana":
            addr = solana_address(rng.randbytes(32))
        elif chain == "xrpl":
            addr = xrp_classic_address(rng.randbytes(20))
        elif chain == "cardano":
            payment = rng.randbytes(28)
            addr = cardano_address(payment, None if style == "enterprise" else rng.randbytes(28))
        else:
            raise ValueError(f"unknown chain {chain!r}")
        if matches_format(addr, ADDRESS_FORMATS[chain]):
            return addr


def random_tx_hash(chain: str, rng: random.Random) -> str:
    """A transaction id in the chain's display form: 32-byte hex (uppercase on the
    XRP Ledger, 0x-prefixed on Ethereum) or a base58 64-byte Solana signature."""
    while True:
        if chain == "solana":
            tx = base58_encode(rng.randbytes(64), BASE58_BITCOIN)
        else:
            digest = rng.randbytes(32).hex()
            if chain == "ethereum":
                tx = "0x" + digest
            elif chain == "xrpl":
                tx = digest.upper()
            elif chain in ("bitcoin", "cardano"):
                tx = digest
            else:
                raise ValueError(f"unknown chain {chain!r}")
        if matches_format(tx, TX_HASH_FORMATS[chain]):
            return tx


def verify_address(chain: str, address: str) -> bool:
    """True when the address decodes with a valid checksum (or, for the two
    checksum-free forms, decodes to the right length) and matches the casefile format."""
    if not matches_format(address, ADDRESS_FORMATS[chain]):
        return False
    try:
        if chain == "bitcoin":
            version, program = decode_segwit(address)
            return version == 0 and len(program) == 20
        if chain == "cardano":
            decode_cardano(address)
            return True
        if chain == "xrpl":
            decode_xrp(address)
            return True
        if chain == "solana":
            return len(base58_decode(address)) == 32
        if chain == "ethereum":
            return address == address.lower() and len(bytes.fromhex(address[2:])) == 20
    except ValueError:
        return False
    return False
