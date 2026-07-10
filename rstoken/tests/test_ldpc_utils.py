import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "13_eval_ldpc_protected.py"
spec = importlib.util.spec_from_file_location("eval_ldpc", SCRIPT)
ldpc = importlib.util.module_from_spec(spec)
sys.modules["eval_ldpc"] = ldpc
spec.loader.exec_module(ldpc)


class TestLDPCUtilities(unittest.TestCase):
    def test_systematic_ldpc_round_trip_without_noise(self):
        code = ldpc.SystematicLDPC(source_bits=64, rate_num=1, rate_den=2, col_weight=3, seed=7)
        bits = ldpc.bytes_to_bits(bytes(range(8)))
        encoded = code.encode(bits)
        llr = ldpc.bits_to_llr(encoded, error_probability=1e-6)
        decoded, success, iterations = code.decode(llr, max_iter=20)
        self.assertTrue(success)
        self.assertLessEqual(iterations, 2)
        self.assertEqual(decoded.tolist(), bits.tolist())

    def test_payload_pack_preserves_length_and_padding(self):
        payload = b"hello world"
        packed = ldpc.pack_payload_bits(payload, source_bits=128)
        restored = ldpc.unpack_payload_bits(packed.bits, packed.original_num_bytes)
        self.assertEqual(restored, payload)
        self.assertEqual(packed.pad_bits, 40)

    def test_payload_pack_rejects_oversize_payload(self):
        with self.assertRaises(ValueError):
            ldpc.pack_payload_bits(b"too long", source_bits=8)

    def test_bpsk_llr_no_channel_keeps_codeword(self):
        codeword = ldpc.bytes_to_bits(b"\x00\xff")
        llr, raw_ber = ldpc.transmit_bpsk_llr(codeword, "none", "inf", seed=1)
        hard = (llr < 0).astype("uint8")
        self.assertEqual(raw_ber, 0.0)
        self.assertEqual(hard.tolist(), codeword.tolist())


if __name__ == "__main__":
    unittest.main()

