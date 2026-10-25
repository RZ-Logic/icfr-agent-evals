"""Address and tx-hash encoders: published vectors, checksum round trips, and the
format and alphabet rules section 6's narrative sweep relies on."""

import random

import pytest
from casefactory import TX_HASHES, WALLETS

from icfr_agent_evals.casefile import ADDRESS_FORMATS, BASE58_XRP, TX_HASH_FORMATS, matches_format
from icfr_agent_evals.chainformats import (
    base58_decode,
    base58_encode,
    bech32_decode,
    decode_cardano,
    decode_segwit,
    decode_xrp,
    p2wpkh_address,
    random_address,
    random_tx_hash,
    verify_address,
    xrp_classic_address,
)

CHAINS = list(ADDRESS_FORMATS)


def test_bip173_p2wpkh_vector():
    # BIP-173: witness v0 program 751e76e8199196d454941c45d1b3a323f1433bd6
    program = bytes.fromhex("751e76e8199196d454941c45d1b3a323f1433bd6")
    assert p2wpkh_address(program) == "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
    assert decode_segwit("BC1QW508D6QEJXTDG4Y5R3ZARVARY0C5XW7KV8F3T4") == (0, program)


@pytest.mark.parametrize(
    "valid",
    ["A12UEL5L", "abcdef1qpzry9x8gf2tvdw0s3jn54khce6mua7lmqqqxw", "split1checkupstagehandshakeupstreamerranterredcaperred2y9e3w"],
)
def test_bip173_valid_checksums(valid):
    bech32_decode(valid)


def test_bech32_rejects_a_flipped_character():
    good = "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
    bad = good[:-1] + ("5" if good[-1] != "5" else "6")
    with pytest.raises(ValueError):
        bech32_decode(bad)


def test_xrp_published_addresses():
    # ACCOUNT_ZERO, ACCOUNT_ONE and the genesis account of the XRP Ledger
    assert xrp_classic_address(bytes(20)) == "rrrrrrrrrrrrrrrrrrrrrhoLvTp"
    assert xrp_classic_address(bytes(19) + b"\x01") == "rrrrrrrrrrrrrrrrrrrrBZbvji"
    genesis = bytes.fromhex("B5F762798A53D543A014CAF8B297CFF8F2F937E8")
    assert xrp_classic_address(genesis) == "rHb9CJAWyB4rj91VRWn96DkukG4bwdtyTh"
    assert decode_xrp("rHb9CJAWyB4rj91VRWn96DkukG4bwdtyTh") == genesis


def test_xrp_rejects_bad_checksum():
    with pytest.raises(ValueError):
        decode_xrp("rHb9CJAWyB4rj91VRWn96DkukG4bwdtyTi")


def test_base58_vector_and_round_trip():
    assert base58_encode(b"Hello World!") == "2NEpo7TZRRrLZSi2U"
    assert base58_decode("11111111111111111111111111111111") == bytes(32)  # Solana system program


def test_cip19_cardano_vectors():
    base = (
        "addr1qx2fxv2umyhttkxyxp8x0dlpdt3k6cwng5pxj3jhsydzer3n0d3vllmyqwsx5wktcd8cc3sq835lu7drv2xwl2wywfgse35a3x"
    )
    enterprise = "addr1vx2fxv2umyhttkxyxp8x0dlpdt3k6cwng5pxj3jhsydzers66hrl8"
    raw_base, raw_ent = decode_cardano(base), decode_cardano(enterprise)
    assert raw_base[0] == 0x01 and raw_ent[0] == 0x61
    assert raw_base[1:29] == raw_ent[1:29]  # same payment key hash in both CIP-19 vectors


@pytest.mark.parametrize("chain", CHAINS)
def test_casefactory_wallets_verify(chain):
    for addr in WALLETS[chain]:
        assert verify_address(chain, addr)
    assert matches_format(TX_HASHES[chain], TX_HASH_FORMATS[chain])


@pytest.mark.parametrize("chain", CHAINS)
def test_random_addresses_verify_and_match_formats(chain):
    rng = random.Random(7)
    for _ in range(300):
        for style in ("default", "enterprise"):
            addr = random_address(chain, rng, style=style)
            assert verify_address(chain, addr), addr
            assert matches_format(addr, ADDRESS_FORMATS[chain])
            if chain == "xrpl":
                assert not set(addr) & set("0OIl")
                assert set(addr) <= set(BASE58_XRP)
            if chain == "cardano":
                assert addr.startswith("addr1") and not addr[5:].isdigit()
                assert not all(c in "0123456789abcdef" for c in addr[5:])
            if chain == "ethereum":
                assert addr == addr.lower()


@pytest.mark.parametrize("chain", CHAINS)
def test_random_tx_hashes_match_formats(chain):
    rng = random.Random(11)
    for _ in range(300):
        assert matches_format(random_tx_hash(chain, rng), TX_HASH_FORMATS[chain])


def test_verify_rejects_tampered_addresses():
    rng = random.Random(3)
    for chain in ("bitcoin", "cardano", "xrpl"):
        addr = random_address(chain, rng)
        alphabet = "qpzry9x8gf2tvdw0s3jn54khce6mua7l" if chain != "xrpl" else BASE58_XRP
        swap = alphabet[(alphabet.index(addr[-2]) + 1) % len(alphabet)]
        assert not verify_address(chain, addr[:-2] + swap + addr[-1])
