#!/usr/bin/env python3
"""
Tests for vLLM execution with PearlMiner mining control.

These tests verify that:
1. vLLM can be initialized with PearlMiner support
2. Mining can be controlled (enabled/disabled) globally
3. The model generates outputs correctly in both mining and non-mining modes
4. VLLM calls work exactly as in the reference mining example
"""

import asyncio
import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest
import torch
from miner_base.settings import MinerSettings
from vllm import LLM, SamplingParams
from vllm_miner.mining_state import get_async_manager, init_async_manager

# Test configuration
TEST_MODEL = "pearl-ai/Llama-3.1-8B-Instruct-pearl"
TEST_MAX_MODEL_LEN = 2048
TEST_PROMPT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are a helpful assistant.<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n"
    "Explain *useful proof of work* simply<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)

# Path to reference outputs for regression testing
REFERENCE_OUTPUTS_FILE = Path(__file__).parent / "reference_outputs.json"

# Set to True to regenerate reference outputs (for initial creation or updates)
REGENERATE_REFERENCES = os.getenv("REGENERATE_VLLM_REFERENCES", "false").lower() in (
    "true",
    "1",
    "yes",
)


class ReferenceOutputManager:
    """Manages reference outputs for regression testing."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.outputs: dict = {}
        self.modified = False
        self._load()

    def _load(self):
        """Load reference outputs from file if it exists."""
        if self.filepath.exists():
            with open(self.filepath) as f:
                self.outputs = json.load(f)

    def save(self):
        """Save reference outputs to file."""
        if self.modified:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, "w") as f:
                json.dump(self.outputs, f, indent=2)
            print(f"\nReference outputs saved to {self.filepath}")

    def validate_or_record(self, test_name: str, actual_output: str) -> bool:
        """
        Validate output against reference or record it if regenerating.

        Returns:
            True if validation passed or output was recorded
        """
        if REGENERATE_REFERENCES:
            self.outputs[test_name] = actual_output
            self.modified = True
            print(f"\nRecorded reference output for {test_name}")
            return True

        if test_name not in self.outputs:
            pytest.fail(
                f"No reference output found for test '{test_name}'.\n"
                f"Run with REGENERATE_VLLM_REFERENCES=true to generate reference outputs."
            )

        reference = self.outputs[test_name]

        # Exact match validation
        if actual_output == reference:
            return True

        # If not exact match, provide detailed error
        pytest.fail(
            f"Output mismatch for test '{test_name}':\n"
            f"Expected length: {len(reference)}\n"
            f"Actual length: {len(actual_output)}\n"
            f"First 200 chars of reference: {reference[:200]}\n"
            f"First 200 chars of actual: {actual_output[:200]}\n"
            f"Run with REGENERATE_VLLM_REFERENCES=true to update reference outputs."
        )


@pytest.fixture(scope="session")
def reference_outputs():
    """Fixture for managing reference outputs."""
    manager = ReferenceOutputManager(REFERENCE_OUTPUTS_FILE)
    yield manager
    manager.save()


def cleanup_llm(llm):
    """Shut down an LLM's engine-core subprocess so GPU memory is released."""
    if llm is None:
        return
    with contextlib.suppress(Exception):
        get_async_manager().wait_until_done_submitting_blocks()
    # vLLM v1's LLM has no __del__; we must terminate the subprocess explicitly.
    with contextlib.suppress(Exception):
        llm.llm_engine.engine_core.shutdown()


def create_llm_instance(tmp_path_factory, no_mining: bool):
    """
    Create a new LLM instance with specified mining configuration.
    """
    # Set environment variables BEFORE creating LLM
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["MINER_DEBUG"] = "true"
    os.environ["MINER_HARD_INNER_HASH"] = "true"
    os.environ["MINER_NO_MINING"] = "true" if no_mining else "false"
    os.environ["MINER_NO_GATEWAY"] = "false"
    os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
    os.environ["PEARL_LOG_LEVEL"] = "DEBUG"

    cache_key = "no_mining" if no_mining else "with_mining"
    tmp_dir = tmp_path_factory.mktemp(f"vllm_logs_{cache_key}")
    log_file = tmp_dir / "vllm_init.log"
    print(f"Logging to {log_file}")

    print(f"\n🚀 Creating LLM instance with MINER_NO_MINING={no_mining}")
    print(f"   Logging to {log_file}")

    # Redirect file descriptors at OS level to capture subprocess output
    old_stdout_fd = os.dup(1)  # Save original stdout
    old_stderr_fd = os.dup(2)  # Save original stderr

    with open(log_file, "w", buffering=1) as log_f:
        # Redirect both stdout and stderr to the log file at OS level
        os.dup2(log_f.fileno(), 1)
        os.dup2(log_f.fileno(), 2)

        try:
            # the plugin is initialized via vLLM's plugin mechanism, see pyproject.toml

            # Initialize LLM with PearlMiner support
            llm = LLM(
                model=TEST_MODEL,
                max_model_len=TEST_MAX_MODEL_LEN,
                enforce_eager=True,
                gpu_memory_utilization=0.9,  # We want to allow 2 LLM instances
            )
        finally:
            # Flush and restore original file descriptors
            sys.stdout.flush()
            sys.stderr.flush()
            os.fsync(1)
            os.fsync(2)

            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

    # Store metadata as an attribute on the llm object
    llm._init_log_file = str(log_file)
    llm._no_mining = no_mining

    print("✅ LLM instance created")

    return llm


@pytest.fixture
def get_llm_instance(tmp_path_factory):
    """
    Fixture that creates LLM instances and automatically cleans them up after the test.

    For tests with one LLM: cleanup is automatic
    For tests with multiple LLMs: call cleanup_llm() between instances, final cleanup is automatic
    """
    created_llms = []

    def _get_llm_instance(is_mining_enabled: bool):
        no_mining = not is_mining_enabled
        llm = create_llm_instance(tmp_path_factory, no_mining)
        created_llms.append(llm)
        return llm

    yield _get_llm_instance

    # Automatic cleanup after test
    for llm in created_llms:
        try:
            cleanup_llm(llm)
        except Exception as e:
            print(f"Warning: Error during automatic cleanup: {e}")


@pytest.fixture(autouse=True)
def manage_mining_state(pearl_gateway_process):
    manager = get_async_manager()
    previous_state = manager._conf.no_mining

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    try:
        yield
    finally:
        manager.wait_until_done_submitting_blocks()
        manager._conf.no_mining = previous_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


@pytest.fixture(scope="session", autouse=True)
def pearl_gateway_process(sample_block_template, mining_address):  # noqa: C901
    """
    Start pearl-gateway programmatically once per test session.

    Starts the gateway and populates its work cache with a mock block template
    to enable testing without blockchain node connectivity.
    """
    from pearl_gateway.pearl_gateway import PearlGateway

    # Set mining address environment variable required by PearlConfig
    os.environ["PEARLD_MINING_ADDRESS"] = mining_address

    # Use default socket path from pearl-gateway config
    socket_path = "/tmp/pearlgw.sock"

    # Remove socket if it exists from previous run
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    # Store gateway instance for work cache access
    gateway_instance = None

    # Run gateway in background thread with custom starter that stores instance
    async def start_gateway_with_mock_template():
        nonlocal gateway_instance
        gateway_instance = PearlGateway(debug_mode=True)

        # Start the gateway
        await gateway_instance.start()

        # Populate work cache with mock block template for testing without node
        print("Populating gateway work cache with mock block template...")
        await gateway_instance.work_cache.update_template(sample_block_template)
        print(
            f"✓ Work cache populated with block template at height {sample_block_template.height}"
        )

        # Keep the service running
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            await gateway_instance.stop()

    def run_gateway():
        asyncio.run(start_gateway_with_mock_template())

    gateway_thread = threading.Thread(target=run_gateway, daemon=True)
    gateway_thread.start()

    # Wait for gateway to create socket file
    max_wait = 10  # seconds
    wait_interval = 0.5
    elapsed = 0

    print(f"Waiting for gateway to create socket at: {socket_path}")

    while elapsed < max_wait:
        if os.path.exists(socket_path):
            print(f"Socket created at: {socket_path}")
            break
        time.sleep(wait_interval)
        elapsed += wait_interval

    if not os.path.exists(socket_path):
        raise RuntimeError(f"Gateway socket not created after {max_wait}s")

    # Additional wait for gateway to fully initialize and populate cache
    time.sleep(2)

    # Initialize async manager in main process (reads settings from environment variables)
    # Note: The worker processes will also initialize their own async managers
    test_settings = MinerSettings()
    print(f"\nTest fixture settings: {test_settings}")
    init_async_manager(test_settings)
    manager = get_async_manager()

    gateway_info = {
        "socket_path": socket_path,
        "thread": gateway_thread,
        "gateway": gateway_instance,
    }

    try:
        yield gateway_info
    finally:
        # Wait for any pending submissions
        manager.wait_until_done_submitting_blocks()

        # Cleanup: gateway thread is daemon so will exit with pytest
        # Remove socket file
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        print("\nGateway terminated")


@pytest.fixture
def deterministic_params():
    """Create deterministic sampling parameters for reproducible test outputs."""
    return SamplingParams(temperature=0.0, max_tokens=100, seed=42)


@pytest.mark.parametrize("is_mining_enabled", [True, False])
def test_pearl_plugin_loaded(get_llm_instance, is_mining_enabled):
    """
    Test that the PearlMiner plugin is properly loaded by checking vLLM initialization logs.

    Expected: "PearlScaledMMLinearKernel for CompressedTensorsW8A8Int8"
    """
    llm_instance = get_llm_instance(is_mining_enabled)
    # Read the initialization log file
    log_file = llm_instance._init_log_file
    with open(log_file) as f:
        log_content = f.read()

    # Check for Pearl kernel (plugin loaded)
    pearl_kernel_msg = "Using PearlKernel (mining_enabled=True) for mining layer"

    if pearl_kernel_msg not in log_content:
        # Check if default kernel is being used instead
        default_kernel_msg = "Using CutlassScaledMMLinearKernel for CompressedTensorsW8A8Int8"

        if default_kernel_msg in log_content:
            pytest.fail(
                f"PearlMiner plugin NOT loaded!\nFound: {default_kernel_msg}\nExpected: {pearl_kernel_msg}"
            )
        else:
            pytest.fail(
                f"Could not find kernel selection message in logs.\nLog file: {log_file} ({len(log_content)} bytes)"
            )


def test_vllm_mining_explicitly_enabled(get_llm_instance, deterministic_params, reference_outputs):
    """Test that model generates correct outputs with mining enabled."""
    llm = get_llm_instance(is_mining_enabled=True)

    manager = get_async_manager()

    outputs = llm.generate(TEST_PROMPT, deterministic_params)
    generated_text = outputs[0].outputs[0].text

    # Wait for all blocks to be submitted before assertions
    manager.wait_until_done_submitting_blocks()

    assert len(generated_text) > 0, "Generated text should not be empty"
    reference_outputs.validate_or_record("test_vllm_mining_explicitly_enabled", generated_text)


def test_vllm_mining_disabled(get_llm_instance, deterministic_params, reference_outputs):
    """Test that model generates correct outputs with mining disabled."""
    llm = get_llm_instance(is_mining_enabled=False)

    outputs = llm.generate(TEST_PROMPT, deterministic_params)
    generated_text = outputs[0].outputs[0].text

    assert len(generated_text) > 0, "Generated text should not be empty"
    reference_outputs.validate_or_record("test_vllm_mining_disabled", generated_text)


def test_vllm_consistency_check(
    deterministic_params,
    reference_outputs,
    get_llm_instance,
):
    """Test that both mining modes produce deterministic outputs."""
    manager = get_async_manager()

    simple_prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        "What is 2+2?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    # Phase 1: Test with mining enabled
    print("\n📝 Phase 1: Testing with mining ENABLED")
    llm_enabled = get_llm_instance(is_mining_enabled=True)

    text_with_mining = llm_enabled.generate(simple_prompt, deterministic_params)[0].outputs[0].text

    # Wait for all blocks to be submitted
    manager.wait_until_done_submitting_blocks()

    assert len(text_with_mining) > 0, "Mining mode should produce output"
    reference_outputs.validate_or_record(
        "test_vllm_consistency_check_with_mining", text_with_mining
    )

    # Clean up first LLM to free GPU memory before creating the second one
    cleanup_llm(llm_enabled)

    # Phase 2: Test with mining disabled
    print("\n📝 Phase 2: Testing with mining DISABLED")
    llm_disabled = get_llm_instance(is_mining_enabled=False)

    text_without_mining = (
        llm_disabled.generate(simple_prompt, deterministic_params)[0].outputs[0].text
    )

    # Wait to ensure no async submissions pending
    manager.wait_until_done_submitting_blocks()

    assert len(text_without_mining) > 0, "Non-mining mode should produce output"
    reference_outputs.validate_or_record(
        "test_vllm_consistency_check_without_mining", text_without_mining
    )
