import time

import pytest
import torch
from pearl_gemm import quantize

DEVICE = "cuda"


def assert_quantization_close(
    output_tensor, scales_tensor, output_ref, scales_ref, fast_math=False
):
    """
    Assert that quantization results match reference within tolerance.

    Args:
        output_tensor: Actual quantized output (int8)
        scales_tensor: Actual scales (float32)
        output_ref: Reference quantized output (int8)
        scales_ref: Reference scales (float32)
        fast_math: If True, allow ±1 rounding difference in output values
    """
    torch.testing.assert_close(scales_tensor.cpu(), scales_ref.cpu(), atol=1e-6, rtol=1e-5)
    atol = 1.0 if fast_math else 0.0  # fast_math may have rounding differences
    torch.testing.assert_close(
        output_tensor.cpu().float(), output_ref.cpu().float(), atol=atol, rtol=0.0
    )


def compute_ref_quantization(x, max_val=63, smooth_scale=None):
    """
    Reference torch implementation supporting all modes.

    Args:
        x: Input tensor (num_tokens, hidden_size)
        max_val: Maximum quantization value (63 for 7-bit, 127 for 8-bit)
        smooth_scale: Optional per-channel scale (hidden_size,) or (1, hidden_size)
            NOTE: smooth_scale contains reciprocal values, so we multiply (not divide)

    Returns:
        xq: Quantized int8 tensor
        xq_scales: Per-token scales (num_tokens, 1), fp32
    """
    x = x.to(torch.float32)
    if smooth_scale is not None:
        smooth_scale = smooth_scale.to(torch.float32)
        # Ensure smooth_scale broadcasts correctly: (hidden_size,) -> (1, hidden_size)
        if smooth_scale.dim() == 1:
            smooth_scale = smooth_scale.unsqueeze(0)
        # smooth_scale contains reciprocals, so multiply instead of divide
        x = x * smooth_scale

    row_max = x.abs().max(dim=-1, keepdim=True).values
    is_zero = row_max == 0
    xq_scales = torch.where(is_zero, torch.zeros_like(row_max), row_max / float(max_val))

    safe_scales = torch.where(is_zero, torch.ones_like(xq_scales), xq_scales)
    xq = (x / safe_scales).round()
    xq = torch.where(is_zero, torch.zeros_like(xq), xq)
    return xq.to(torch.int8), xq_scales


# ===== Fixtures =====


@pytest.fixture(autouse=True)
def seed():
    """Set random seed for reproducibility."""
    torch.random.manual_seed(42)


@pytest.fixture(autouse=True)
def clear_gpu_cache():
    """Clear GPU cache after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def make_input():
    """Factory fixture for creating random input tensors in [-scale, scale]."""

    def _make_input(num_tokens, hidden_size, dtype=torch.float16, scale=1.0):
        return torch.rand(num_tokens, hidden_size, dtype=dtype, device=DEVICE) * 2 * scale - scale

    return _make_input


@pytest.fixture
def make_output_tensors():
    """Factory fixture for creating output and scales tensors."""

    def _make_output_tensors(num_tokens, hidden_size):
        output = torch.empty(num_tokens, hidden_size, dtype=torch.int8, device=DEVICE)
        scales = torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE)
        return output, scales

    return _make_output_tensors


@pytest.fixture
def make_smooth_scale():
    """Factory fixture for creating smooth scale tensors in [low, high]."""

    def _make_smooth_scale(hidden_size, low=0.5, high=2.0, dtype=torch.float32):
        return torch.rand(hidden_size, dtype=dtype, device=DEVICE) * (high - low) + low

    return _make_smooth_scale


class TestQuantizationCore:
    """Core functionality tests for dynamic per-token quantization."""

    @pytest.mark.parametrize("num_tokens", [1, 4, 7, 15, 16, 33, 64, 512])
    @pytest.mark.parametrize("hidden_size", [127, 128, 255, 512, 1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("max_val", [63, 127])
    @pytest.mark.parametrize("fast_math", [False, True])
    @pytest.mark.parametrize(
        "smooth_scale_dtype", [None, torch.float32, torch.float16, torch.bfloat16]
    )
    def test_quantize(
        self,
        num_tokens,
        hidden_size,
        dtype,
        max_val,
        fast_math,
        smooth_scale_dtype,
        make_input,
        make_output_tensors,
        make_smooth_scale,
    ):
        """Test quantization with different smooth scale dtypes (None, float32, same as input)"""
        input_tensor = make_input(num_tokens, hidden_size, dtype=dtype)
        smooth_scale = (
            None
            if smooth_scale_dtype is None
            else make_smooth_scale(hidden_size, dtype=smooth_scale_dtype)
        )
        output_tensor, scales_tensor = make_output_tensors(num_tokens, hidden_size)

        # Run quantization
        quantize(
            input_tensor,
            output_tensor,
            scales_tensor,
            max_val=max_val,
            smooth_scale=smooth_scale,
            fast_math=fast_math,
        )

        # Compute reference
        output_ref, scales_ref = compute_ref_quantization(
            input_tensor, max_val=max_val, smooth_scale=smooth_scale
        )

        # Shape checks
        assert output_tensor.shape == input_tensor.shape
        assert scales_tensor.shape == (num_tokens, 1)

        # Range checks
        assert torch.all(output_tensor >= -max_val)
        assert torch.all(output_tensor <= max_val)
        assert torch.all(scales_tensor >= 0)

        # Value checks
        assert_quantization_close(
            output_tensor, scales_tensor, output_ref, scales_ref, fast_math=fast_math
        )


class TestSmoothScale:
    """Tests specific to smooth scale functionality."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("max_val", [63, 127])
    @pytest.mark.parametrize("smooth_scale_dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_identity(self, dtype, max_val, smooth_scale_dtype, make_input, make_output_tensors):
        """Smooth scale of 1.0 should match no smooth scale"""
        num_tokens, hidden_size = 32, 512

        input_tensor = make_input(num_tokens, hidden_size, dtype=dtype)
        smooth_scale = torch.ones(hidden_size, dtype=smooth_scale_dtype, device=DEVICE)

        # With smooth scale = 1.0
        output_with_scale, scales_with_scale = make_output_tensors(num_tokens, hidden_size)
        quantize(
            input_tensor,
            output_with_scale,
            scales_with_scale,
            max_val=max_val,
            smooth_scale=smooth_scale,
        )

        # Without smooth scale
        output_without_scale, scales_without_scale = make_output_tensors(num_tokens, hidden_size)
        quantize(
            input_tensor,
            output_without_scale,
            scales_without_scale,
            max_val=max_val,
            smooth_scale=None,
        )

        # Should be identical
        torch.testing.assert_close(output_with_scale, output_without_scale)
        torch.testing.assert_close(scales_with_scale, scales_without_scale)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("max_val", [63, 127])
    @pytest.mark.parametrize(
        "scale_type", ["uniform_0.5", "uniform_2.0", "uniform_10.0", "varying"]
    )
    @pytest.mark.parametrize("smooth_scale_dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_patterns(
        self, dtype, max_val, scale_type, smooth_scale_dtype, make_input, make_output_tensors
    ):
        """Test uniform and varying smooth scale patterns with different dtypes"""
        num_tokens, hidden_size = 32, 512

        input_tensor = make_input(num_tokens, hidden_size, dtype=dtype)

        # Create smooth scale based on pattern type
        if scale_type.startswith("uniform_"):
            scale_value = float(scale_type.split("_")[1])
            smooth_scale = torch.full(
                (hidden_size,), scale_value, dtype=smooth_scale_dtype, device=DEVICE
            )
        else:  # varying
            smooth_scale = torch.linspace(
                0.1, 10.0, hidden_size, dtype=smooth_scale_dtype, device=DEVICE
            )

        output_tensor, scales_tensor = make_output_tensors(num_tokens, hidden_size)

        quantize(
            input_tensor, output_tensor, scales_tensor, max_val=max_val, smooth_scale=smooth_scale
        )

        # Compute reference
        output_ref, scales_ref = compute_ref_quantization(
            input_tensor, max_val=max_val, smooth_scale=smooth_scale
        )

        assert_quantization_close(output_tensor, scales_tensor, output_ref, scales_ref)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("max_val", [63, 127])
    @pytest.mark.parametrize(
        "scale_value,input_scale",
        [
            (1000.0, 100.0),  # Large smooth scale with moderate input
            (0.001, 0.01),  # Small smooth scale with small input
        ],
    )
    @pytest.mark.parametrize("smooth_scale_dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_extreme_values(
        self,
        dtype,
        max_val,
        scale_value,
        input_scale,
        smooth_scale_dtype,
        make_input,
        make_output_tensors,
    ):
        """Test extreme smooth scale values (very large or very small) with different dtypes"""
        num_tokens, hidden_size = 8, 256

        input_tensor = torch.rand(num_tokens, hidden_size, dtype=dtype, device=DEVICE) * input_scale
        smooth_scale = torch.full(
            (hidden_size,), scale_value, dtype=smooth_scale_dtype, device=DEVICE
        )
        output_tensor, scales_tensor = make_output_tensors(num_tokens, hidden_size)

        quantize(
            input_tensor, output_tensor, scales_tensor, max_val=max_val, smooth_scale=smooth_scale
        )

        # Compute reference
        output_ref, scales_ref = compute_ref_quantization(
            input_tensor, max_val=max_val, smooth_scale=smooth_scale
        )

        # Range check
        assert torch.all(output_tensor >= -max_val)
        assert torch.all(output_tensor <= max_val)

        assert_quantization_close(output_tensor, scales_tensor, output_ref, scales_ref)


class TestQuantizationEdgeCases:
    """Edge case tests for quantization."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("max_val", [63, 127])
    @pytest.mark.parametrize(
        "smooth_scale_dtype", [None, torch.float32, torch.float16, torch.bfloat16]
    )
    def test_zero_input(
        self, dtype, max_val, smooth_scale_dtype, make_output_tensors, make_smooth_scale
    ):
        """Zero input rows should produce zero output and zero scale"""
        num_tokens, hidden_size = 4, 256

        input_tensor = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=DEVICE)
        smooth_scale = (
            None
            if smooth_scale_dtype is None
            else make_smooth_scale(hidden_size, dtype=smooth_scale_dtype)
        )
        output_tensor, scales_tensor = make_output_tensors(num_tokens, hidden_size)

        quantize(
            input_tensor, output_tensor, scales_tensor, max_val=max_val, smooth_scale=smooth_scale
        )

        assert torch.all(output_tensor == 0)
        assert torch.all(scales_tensor == 0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("max_val", [63, 127])
    @pytest.mark.parametrize("smooth_scale_dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_mixed_zero_and_nonzero_rows(
        self, dtype, max_val, smooth_scale_dtype, make_input, make_output_tensors, make_smooth_scale
    ):
        """Mix of zero and non-zero rows"""
        num_tokens, hidden_size = 8, 256

        input_tensor = make_input(num_tokens, hidden_size, dtype=dtype)
        # Zero out some rows
        input_tensor[1, :] = 0
        input_tensor[3, :] = 0
        input_tensor[7, :] = 0

        smooth_scale = make_smooth_scale(hidden_size, dtype=smooth_scale_dtype)
        output_tensor, scales_tensor = make_output_tensors(num_tokens, hidden_size)

        quantize(
            input_tensor, output_tensor, scales_tensor, max_val=max_val, smooth_scale=smooth_scale
        )

        # Compute reference
        output_ref, scales_ref = compute_ref_quantization(
            input_tensor, max_val=max_val, smooth_scale=smooth_scale
        )

        # Check zero rows
        assert torch.all(output_tensor[1, :] == 0)
        assert torch.all(output_tensor[3, :] == 0)
        assert torch.all(output_tensor[7, :] == 0)
        assert scales_tensor[1, 0] == 0
        assert scales_tensor[3, 0] == 0
        assert scales_tensor[7, 0] == 0

        assert_quantization_close(output_tensor, scales_tensor, output_ref, scales_ref)


class TestQuantizationRangePreservation:
    """Tests for output range preservation."""

    @pytest.mark.parametrize("max_val", [63, 127])
    @pytest.mark.parametrize("use_smooth_scale", [False, True])
    @pytest.mark.parametrize("input_type", ["large_dynamic_range", "extreme_values"])
    def test_output_range_preservation(
        self, max_val, use_smooth_scale, input_type, make_output_tensors, make_smooth_scale
    ):
        """Output values must be in [-max_val, max_val] for various input patterns"""
        dtype = torch.float16

        if input_type == "large_dynamic_range":
            num_tokens, hidden_size = 64, 1024
            input_tensor = torch.randn(num_tokens, hidden_size, dtype=dtype, device=DEVICE) * 1000
        else:  # extreme_values
            num_tokens, hidden_size = 4, 128
            # Create input with extreme values (near fp16 max)
            input_tensor = torch.tensor(
                [
                    [65000.0, -65000.0] + [0.0] * (hidden_size - 2),
                    [1.0, -1.0] + [0.5] * (hidden_size - 2),
                    [0.001, -0.001] + [0.0005] * (hidden_size - 2),
                    [100.0, 100.0] + [100.0] * (hidden_size - 2),
                ],
                dtype=dtype,
                device=DEVICE,
            )

        smooth_scale = (
            make_smooth_scale(hidden_size, low=0.1, high=10.0) if use_smooth_scale else None
        )
        output_tensor, scales_tensor = make_output_tensors(num_tokens, hidden_size)

        quantize(
            input_tensor, output_tensor, scales_tensor, max_val=max_val, smooth_scale=smooth_scale
        )

        # Strict range check
        assert torch.all(output_tensor >= -max_val), f"Found values below -{max_val}"
        assert torch.all(output_tensor <= max_val), f"Found values above {max_val}"
        assert torch.all(scales_tensor >= 0)


class TestQuantizationInputValidation:
    """Input validation tests for quantization."""

    def test_invalid_input_dtype(self):
        """Input must be float16 or bfloat16"""
        input_tensor = torch.randn(4, 128, dtype=torch.float32, device=DEVICE)
        output_tensor = torch.empty(4, 128, dtype=torch.int8, device=DEVICE)
        scales_tensor = torch.empty(4, 1, dtype=torch.float32, device=DEVICE)

        with pytest.raises(RuntimeError, match="Input must be float16 or bfloat16"):
            quantize(input_tensor, output_tensor, scales_tensor)

    def test_invalid_output_dtype(self):
        """Output must be int8"""
        input_tensor = torch.randn(4, 128, dtype=torch.float16, device=DEVICE)
        output_tensor = torch.empty(4, 128, dtype=torch.float16, device=DEVICE)
        scales_tensor = torch.empty(4, 1, dtype=torch.float32, device=DEVICE)

        with pytest.raises(RuntimeError, match="Output must be int8"):
            quantize(input_tensor, output_tensor, scales_tensor)

    def test_invalid_scales_dtype(self):
        """Scales must be float32"""
        input_tensor = torch.randn(4, 128, dtype=torch.float16, device=DEVICE)
        output_tensor = torch.empty(4, 128, dtype=torch.int8, device=DEVICE)
        scales_tensor = torch.empty(4, 1, dtype=torch.float16, device=DEVICE)

        with pytest.raises(RuntimeError, match="Scales must be float32"):
            quantize(input_tensor, output_tensor, scales_tensor)

    def test_invalid_smooth_scale_dtype(self):
        """Smooth scale must be float32, float16, or bfloat16"""
        input_tensor = torch.randn(4, 128, dtype=torch.float16, device=DEVICE)
        output_tensor = torch.empty(4, 128, dtype=torch.int8, device=DEVICE)
        scales_tensor = torch.empty(4, 1, dtype=torch.float32, device=DEVICE)
        # int8 smooth_scale should fail
        smooth_scale = torch.ones(128, dtype=torch.int8, device=DEVICE)

        with pytest.raises(
            RuntimeError, match="Smooth scale must be float32, float16, or bfloat16"
        ):
            quantize(input_tensor, output_tensor, scales_tensor, smooth_scale=smooth_scale)

    def test_invalid_smooth_scale_shape(self):
        """Smooth scale shape must match hidden_size"""
        input_tensor = torch.randn(4, 128, dtype=torch.float16, device=DEVICE)
        output_tensor = torch.empty(4, 128, dtype=torch.int8, device=DEVICE)
        scales_tensor = torch.empty(4, 1, dtype=torch.float32, device=DEVICE)
        smooth_scale = torch.ones(64, dtype=torch.float32, device=DEVICE)  # Wrong size

        with pytest.raises(RuntimeError, match="Smooth scale size must match hidden_size"):
            quantize(input_tensor, output_tensor, scales_tensor, smooth_scale=smooth_scale)

    def test_invalid_max_val(self):
        """max_val must be 63 or 127"""
        input_tensor = torch.randn(4, 128, dtype=torch.float16, device=DEVICE)
        output_tensor = torch.empty(4, 128, dtype=torch.int8, device=DEVICE)
        scales_tensor = torch.empty(4, 1, dtype=torch.float32, device=DEVICE)

        with pytest.raises(RuntimeError, match="max_val must be 63 or 127"):
            quantize(input_tensor, output_tensor, scales_tensor, max_val=100)

    def test_invalid_shape_mismatch(self):
        """Input and output shapes must match"""
        input_tensor = torch.randn(4, 128, dtype=torch.float16, device=DEVICE)
        output_tensor = torch.empty(4, 64, dtype=torch.int8, device=DEVICE)  # Wrong size
        scales_tensor = torch.empty(4, 1, dtype=torch.float32, device=DEVICE)

        with pytest.raises(
            RuntimeError, match=r"output must have shape \(num_tokens, hidden_size\)"
        ):
            quantize(input_tensor, output_tensor, scales_tensor)

    def test_invalid_scales_shape(self):
        """Scales shape must be (num_tokens, 1)"""
        input_tensor = torch.randn(4, 128, dtype=torch.float16, device=DEVICE)
        output_tensor = torch.empty(4, 128, dtype=torch.int8, device=DEVICE)
        scales_tensor = torch.empty(8, 1, dtype=torch.float32, device=DEVICE)  # Wrong num_tokens

        with pytest.raises(RuntimeError, match=r"scales must have shape \(num_tokens, 1\)"):
            quantize(input_tensor, output_tensor, scales_tensor)


class TestQuantizationConsistency:
    """Consistency and determinism tests for quantization."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("max_val", [63, 127])
    @pytest.mark.parametrize("smooth_scale_dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_deterministic(self, dtype, max_val, smooth_scale_dtype, make_output_tensors):
        """Quantization should be deterministic"""
        num_tokens, hidden_size = 16, 512

        input_tensor = torch.randn(num_tokens, hidden_size, dtype=dtype, device=DEVICE)
        smooth_scale = torch.rand(hidden_size, dtype=smooth_scale_dtype, device=DEVICE) + 0.5

        results = []
        for _ in range(3):
            output_tensor, scales_tensor = make_output_tensors(num_tokens, hidden_size)

            quantize(
                input_tensor,
                output_tensor,
                scales_tensor,
                max_val=max_val,
                smooth_scale=smooth_scale,
            )

            results.append((output_tensor.clone(), scales_tensor.clone()))

        # All results should be identical
        for i in range(1, len(results)):
            torch.testing.assert_close(results[0][0], results[i][0])
            torch.testing.assert_close(results[0][1], results[i][1])


class TestQuantizationPerformance:
    """Performance benchmarks for quantization."""

    @pytest.mark.slow
    @pytest.mark.parametrize("max_val", [63, 127])
    @pytest.mark.parametrize("use_smooth_scale", [False, True])
    @pytest.mark.flaky(reruns=3)
    def test_performance(self, max_val, use_smooth_scale, make_output_tensors):
        """Benchmark different modes"""
        num_tokens, hidden_size = 1024, 4096
        num_warmup = 10
        num_runs = 100

        input_tensor = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=DEVICE)
        smooth_scale = (
            torch.rand(hidden_size, dtype=torch.float32, device=DEVICE) + 0.5
            if use_smooth_scale
            else None
        )

        # Warmup
        for _ in range(num_warmup):
            output_tensor, scales_tensor = make_output_tensors(num_tokens, hidden_size)
            quantize(
                input_tensor,
                output_tensor,
                scales_tensor,
                max_val=max_val,
                smooth_scale=smooth_scale,
            )

        torch.cuda.synchronize()

        # Benchmark
        start_time = time.perf_counter()
        for _ in range(num_runs):
            output_tensor, scales_tensor = make_output_tensors(num_tokens, hidden_size)
            quantize(
                input_tensor,
                output_tensor,
                scales_tensor,
                max_val=max_val,
                smooth_scale=smooth_scale,
            )

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        avg_time_ms = (end_time - start_time) / num_runs * 1000

        smooth_str = "with" if use_smooth_scale else "without"
        print(
            f"\nQuantization (max_val={max_val}, {smooth_str} smooth_scale): {avg_time_ms:.3f} ms"
        )

        # Sanity check: should complete in reasonable time
        assert avg_time_ms < 10.0, f"Quantization too slow: {avg_time_ms:.3f} ms"
