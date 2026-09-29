import numpy as np
import pytest

from wearusfm.tokenizer_checks.neurorvq import stack_codes, tokens_from_forward


def test_tokens_from_forward_same_order_for_both_output_shapes():
    b, n, a = 2, 3, 4
    x = np.arange(b * n * a * 200, dtype=float).reshape(b, n * a, 200)
    xrec_4d = x.reshape(b, n, a, 200).copy()
    tx, tr = tokens_from_forward(x, xrec_4d, n, a)
    assert tx.shape == tr.shape == (b * n * a, 200)
    assert np.array_equal(tx, tr)  # (b, canale, patch) in entrambi
    tx2, tr2 = tokens_from_forward(x, x.copy(), n, a)
    assert np.array_equal(tx2, tr2)


def test_tokens_from_forward_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        tokens_from_forward(np.zeros((2, 5, 200)), np.zeros((2, 5, 200)), 3, 4)
    with pytest.raises(ValueError):
        tokens_from_forward(np.zeros((2, 12, 200)), np.zeros((2, 11, 200)), 3, 4)


def test_stack_codes_shape_check():
    ok = [np.zeros((16, 10), dtype=np.int64) for _ in range(4)]
    out = stack_codes(ok, 10)
    assert out.shape == (4, 16, 10) and out.dtype == np.int32
    with pytest.raises(ValueError, match="forma inattesa"):
        stack_codes([np.zeros((8, 10)) for _ in range(4)], 10)  # il .view(8) del repo darebbe questo
