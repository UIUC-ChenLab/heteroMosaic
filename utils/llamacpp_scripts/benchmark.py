import argparse
import datetime
import json
import os
import subprocess
import glob
import re
import math
import signal
import time
import select
import pty
import sys

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
BENCHMARK_DIR = os.path.join(REPO_ROOT, "build", "benchmarks", "llama.cpp")
os.makedirs(BENCHMARK_DIR, exist_ok=True)
STATE_FILE = os.path.join(BENCHMARK_DIR, "benchmark.json")
PROMPTS_FILE = "../../py/prompts.txt"
PROJECT_BUILD_DIR = "../../build"
MODELS_DIR = f"{PROJECT_BUILD_DIR}/models"
DEFAULT_PROMPT_SIZES = [256, 512, 1024, 2048, 4096, 8196, 16384]
COOLDOWN_SEC = 16
# MICRO_BATCH_SIZE behavior:
#   > 0 : use the value directly
#   -2  : use 4096 if prompt tokens > 4096, else half prompt tokens
#   else: use full prompt tokens
MICRO_BATCH_SIZE = -2
MODEL_PATTERNS = {
    "llama3-8b_q4_k_s": "*llama3-8b*q4_k_s.gguf",
    "llama3-8b-q8_0": "*llama3-8b*q8_0.gguf",
    "phi3.5-3.8b_q4_k_s": "*phi-3.5-mini-instruct*q4_k_s.gguf",
    "qwen3b_q4_k_s": "*qwen2.5-3b*q4_k_s.gguf",
    "qwen2.5-3b-q8_0": "*qwen2.5-3b*q8_0.gguf",
    "qwen2.5-1.5b_q4_k_s": "*qwen2.5-1.5b*q4_k_s.gguf",
    "qwen2.5-14b_q4_k_s": "*qwen2.5-14b*q4_k_s.gguf",
    "qwen2.5-1.5b-q8_0": "*qwen2.5-1.5b*q8_0.gguf",
    "qwen2.5-14b-q8_0": "*qwen2.5-14b*q8_0.gguf",
    "gemma1-2b_q4_k_s": "*gemma-2b*q4_k_s.gguf",
    "gemma1-2b_q8_0": "*gemma-2b-it*q8_0.gguf",
    "llama3-70b_q4_k_s": "*llama3-70b*q4_k_s.gguf",
    "mixtral-8x7b_Q4_K_M": "*mixtral-8x7b*Q4_K_M.gguf",
}
DEFAULT_MODELS = ["llama3-8b-q8_0"]

MODEL_NAME_ALIASES = {
    "llama3-8b": "llama3-8b_q4_k_s",
    "phi3.5-3.8b": "phi3.5-3.8b_q4_k_s",
    "qwen3b": "qwen3b_q4_k_s",
    "qwen2.5-1.5b": "qwen2.5-1.5b_q4_k_s",
    "qwen2.5-14b": "qwen2.5-14b_q4_k_s",
    "gemma1-2b": "gemma1-2b_q4_k_s",
    "gemma1-2b-q8_0": "gemma1-2b_q8_0",
    "llama3-70b": "llama3-70b_q4_k_s",
    "mixtral-8x7b": "mixtral-8x7b_Q4_K_M",
}

# Map model names to specific filenames to ensure patterns match after download
MODEL_FILENAMES = {
    "llama3-8b_q4_k_s": "llama3-8b-instruct-q4_k_s.gguf",
    "llama3-8b-q8_0": "llama3-8b-instruct-q8_0.gguf",
    "llama3-70b_q4_k_s": "llama3-70b-instruct-q4_k_s.gguf",
    "phi3.5-3.8b_q4_k_s": "phi-3.5-mini-instruct-q4_k_s.gguf",
    "qwen3b_q4_k_s": "qwen2.5-3b-instruct-q4_k_s.gguf",
    "qwen2.5-3b-q8_0": "qwen2.5-3b-instruct-q8_0.gguf",
    "qwen2.5-1.5b_q4_k_s": "qwen2.5-1.5b-instruct-q4_k_s.gguf",
    "qwen2.5-14b_q4_k_s": "qwen2.5-14b-instruct-q4_k_s.gguf",
    "qwen2.5-1.5b-q8_0": "qwen2.5-1.5b-instruct-q8_0.gguf",
    "qwen2.5-14b-q8_0": "qwen2.5-14b-instruct-q8_0.gguf",
    "gemma1-2b_q4_k_s": "gemma-2b-it-q4_k_s.gguf",
    "gemma1-2b_q8_0": "gemma-2b-it-q8_0.gguf",
    "mixtral-8x7b_Q4_K_M": "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
}

# Hugging Face fallback URLs (using Bartowski/Qwen optimized GGUFs)
MODEL_URLS = {
    "llama3-8b_q4_k_s": "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_S.gguf",
    "llama3-8b-q8_0": "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q8_0.gguf",
    "llama3-70b_q4_k_s": "https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct-Q4_K_S.gguf",
    "phi3.5-3.8b_q4_k_s": "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_S.gguf",
    "qwen3b_q4_k_s": "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_S.gguf",
    "qwen2.5-3b-q8_0": "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q8_0.gguf",
    "qwen2.5-1.5b_q4_k_s": "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_S.gguf",
    "qwen2.5-14b_q4_k_s": "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_S.gguf",
    "qwen2.5-1.5b-q8_0": "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q8_0.gguf",
    "qwen2.5-14b-q8_0": "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q8_0.gguf",
    "gemma1-2b_q4_k_s": "https://huggingface.co/mlabonne/gemma-2b-it-GGUF/resolve/main/gemma-2b-it.Q4_K_S.gguf",
    "gemma1-2b_q8_0": "https://huggingface.co/mlabonne/gemma-2b-it-GGUF/resolve/main/gemma-2b-it.Q8_0.gguf",
}

# Original weights configuration for fallback
ORIGINAL_MODEL_REPOS = {
    "mixtral-8x7b_Q4_K_M": "mistralai/Mixtral-8x7B-Instruct-v0.1"
}

# Search path for models
SEARCH_PATHS = [MODELS_DIR]

# Binary path configuration
# Set USE_OLD_BUILD = True to use the old llama.cpp version (after running setup_old.sh)
USE_OLD_BUILD = False
LLAMA_DIR = f"{PROJECT_BUILD_DIR}/llama.cpp"
LLAMA_BUILD_DIR = f"{LLAMA_DIR}/build"

if USE_OLD_BUILD:
    LLAMA_BIN = "./llama.cpp.old/build/bin/llama-completion"
else:
    LLAMA_BIN = f"{LLAMA_BUILD_DIR}/bin/llama-completion"

def get_hsa_target():
    """Detect appropriate HSA_OVERRIDE_GFX_VERSION based on rocminfo.
    Returns None for native gfx1150/gfx1151 support (no override needed)."""
    try:
        # Check for gfx1150/gfx1151 support in rocminfo (native, no override)
        result = subprocess.run(["rocminfo"], capture_output=True, text=True)
        if result.returncode == 0:
            if "gfx1151" in result.stdout:
                print("  [Setup] Detected gfx1151 GPU, running natively (no HSA override)")
                return None
            if "gfx1150" in result.stdout:
                print("  [Setup] Detected gfx1150 GPU, running natively (no HSA override)")
                return None
    except Exception as e:
        print(f"  [Setup] Warning: rocminfo check failed: {e}")
    
    print("  [Setup] Defaulting to HSA target 11.0.0")
    return "11.0.0"

HSA_TARGET_VERSION = get_hsa_target()

def download_and_convert_model(model_name):
    print(f"Attempting to download and convert {model_name}...")
    
    if model_name != "mixtral-8x7b_Q4_K_M":
        print("Fallback conversion only implemented for Mixtral-8x7b")
        return False
        
    print("Downloading original weights using hf-cli...")
    repo_id = ORIGINAL_MODEL_REPOS[model_name]
    raw_model_dir = os.path.join(
        MODELS_DIR, "raw", "Mixtral-8x7B-Instruct-v0.1"
    )
    
    try:
        subprocess.run([
            "hf", "download", repo_id, 
            "--local-dir", raw_model_dir
        ], check=True)
    except subprocess.CalledProcessError:
        print("Failed to download original weights")
        return False
        
    print("Converting to GGUF (f16)...")
    convert_script = f"{LLAMA_DIR}/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        print(f"Convert script not found at {convert_script}")
        return False
        
    outfile_f16 = os.path.join(
        MODELS_DIR, "mixtral-8x7b-instruct-v0.1.fp16.gguf"
    )
    try:
        subprocess.run([
            sys.executable, convert_script, raw_model_dir,
            "--outfile", outfile_f16,
            "--outtype", "f16"
        ], check=True)
    except subprocess.CalledProcessError:
        print("Failed to convert to GGUF")
        return False
        
    print("Quantizing to Q4_K_M...")
    quantize_bin = f"{LLAMA_BUILD_DIR}/bin/llama-quantize"
    outfile_q4 = os.path.join(
        MODELS_DIR, "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
    )
    
    if not os.path.exists(quantize_bin):
        print(f"Quantize binary not found at {quantize_bin}")
        return False
        
    try:
        subprocess.run([
            quantize_bin, outfile_f16, outfile_q4, "Q4_K_M"
        ], check=True)
    except subprocess.CalledProcessError:
        print("Failed to quantize")
        return False
        
    # Optional: cleanup f16 file to save space?
    # os.remove(outfile_f16)
    
    print(f"Successfully created {outfile_q4}")
    return True

def download_model(model_name, url):
    if not url:
        print(f"No download URL known for {model_name}")
        return None
    
    output_dir = MODELS_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Use preferred filename if available, else derive from URL
    if model_name in MODEL_FILENAMES:
        filename = MODEL_FILENAMES[model_name]
    else:
        filename = url.split('/')[-1]
        
    output_path = os.path.join(output_dir, filename)
    
    print(f"Downloading {model_name} to {output_path}...")
    print(f"  Source: {url}")
    
    try:
        subprocess.run(["wget", "-q", "--show-progress", "-O", output_path, url], check=True)
        print("  Download complete.")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"  Download failed: {e}")
        return None

def find_or_download_model(model_name, pattern):
    # Try to find existing
    for path in SEARCH_PATHS:
        full_pattern = os.path.join(path, pattern)
        matches = glob.glob(full_pattern)
        if matches:
            return matches[0]
            
    # If not found, try to download
    print(f"Model {model_name} ({pattern}) not found in search paths.")
    
    # Try GGUF download first
    if model_name in MODEL_URLS:
        downloaded = download_model(model_name, MODEL_URLS[model_name])
        if downloaded:
            return downloaded
            
    # Try fallback conversion
    if download_and_convert_model(model_name):
        return find_or_download_model(model_name, pattern) # Recursively find the new file
    
    return None

def get_prompt_text(max_tokens):
    # Estimate characters: 1 token ~= 4 chars. 
    # This is rough; for exact token count we'd need a tokenizer.
    # We'll overwrite with a bit more to be safe, but main will truncate if -c is set? 
    # Actually main processes what is given. 
    # We will grab enough text.
    with open(PROMPTS_FILE, 'r') as f:
        text = f.read()
    
    char_estimate = max_tokens * 4
    if char_estimate > len(text):
        # Repeat text if not enough
        multiplier = math.ceil(char_estimate / len(text))
        text = text * multiplier
    
    return text[:char_estimate]

def calibrate_token_count(model_path, prompt_text, target_tokens):
    """
    Use llama-tokenize to count exact tokens and adjust prompt to match target.
    Returns (adjusted_prompt, actual_token_count)
    """
    import tempfile
    
    tokenize_bin = LLAMA_BIN.replace("llama-completion", "llama-tokenize")
    if not os.path.exists(tokenize_bin):
        print(f"  Warning: llama-tokenize not found at {tokenize_bin}, using estimate")
        return prompt_text, target_tokens
    
    # Binary search to find right prompt length for target tokens
    low = len(prompt_text) // 4
    high = len(prompt_text)
    best_prompt = prompt_text
    best_tokens = 0
    
    for _ in range(10):  # Max 10 iterations
        mid = (low + high) // 2
        test_prompt = prompt_text[:mid]
        
        # Write to temp file and tokenize
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_prompt)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [tokenize_bin, "-m", model_path, "-f", temp_path, "--ids"],
                capture_output=True, text=True, timeout=1000
            )
            os.unlink(temp_path)
            
            # Parse token count from output like [1, 2, 3, ...]
            if result.returncode == 0:
                match = re.search(r'\[([0-9, ]+)\]', result.stdout)
                if match:
                    tokens = len(match.group(1).split(','))
                    if tokens == target_tokens:
                        return test_prompt, tokens
                    elif tokens < target_tokens:
                        low = mid + 1
                        if tokens > best_tokens:
                            best_prompt = test_prompt
                            best_tokens = tokens
                    else:
                        high = mid - 1
                        best_prompt = test_prompt
                        best_tokens = tokens
        except Exception as e:
            print(f"  Tokenize error: {e}")
            os.unlink(temp_path)
            break
    
    return best_prompt, best_tokens

def run_calibration_test(model_path, prompt_sizes):
    """Run a quick test to measure actual vs estimated tokens"""
    print("\n=== Token Calibration Test ===")
    
    tokenize_bin = LLAMA_BIN.replace("llama-completion", "llama-tokenize")
    if not os.path.exists(tokenize_bin):
        print(f"  llama-tokenize not found, skipping calibration")
        return {}
    
    results = {}
    for target in prompt_sizes:
        prompt = get_prompt_text(target)
        
        # Write temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(prompt)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [tokenize_bin, "-m", model_path, "-f", temp_path, "--ids"],
                capture_output=True, text=True, timeout=1000
            )
            os.unlink(temp_path)
            
            if result.returncode == 0:
                match = re.search(r'\[([0-9, ]+)\]', result.stdout)
                if match:
                    actual = len(match.group(1).split(','))
                    diff_pct = ((actual - target) / target) * 100
                    results[target] = actual
                    print(f"  Target: {target:5d} tokens | Actual: {actual:5d} | Diff: {diff_pct:+.1f}%")
        except Exception as e:
            print(f"  Error for {target}: {e}")
            try:
                os.unlink(temp_path)
            except:
                pass
    
    print("=== End Calibration ===\n")
    return results

def parse_results(output):
    # Parse llama.cpp output for prompt eval metrics and generation metrics
    metrics = {}
    
    # Format 1: Detailed Table (typically stderr)
    pe_match = re.search(r"prompt eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*[\d\.]+\s*ms per token,\s*([\d\.]+)\s*tokens per second\)", output)
    if pe_match:
        metrics["pp_ms"] = float(pe_match.group(1)) # TTFT
        metrics["pp_tokens"] = int(pe_match.group(2))
        metrics["pp_tps"] = float(pe_match.group(3))
    
    gen_match = re.search(r"eval time\s*=\s*([\d\.]+)\s*ms\s*/\s*(\d+)\s*runs\s*\(\s*[\d\.]+\s*ms per token,\s*([\d\.]+)\s*tokens per second\)", output)
    if gen_match:
        metrics["gen_ms"] = float(gen_match.group(1))
        metrics["gen_tokens"] = int(gen_match.group(2))
        metrics["gen_tps"] = float(gen_match.group(3))

    # Format 2: Simple Output (typically stdout)
    # [ Prompt: 412.4 t/s | Generation: 16.4 t/s ]
    if not metrics:
        simple_match = re.search(r"\[ Prompt: ([\d\.]+) t/s \| Generation: ([\d\.]+) t/s \]", output)
        if simple_match:
            metrics["pp_tps"] = float(simple_match.group(1))
            metrics["gen_tps"] = float(simple_match.group(2))
            # pp_ms/tokens not explicit, logic in main will handle estimates
        
    return metrics if metrics else None


def iso_now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def canonical_model_name(model_name):
    return MODEL_NAME_ALIASES.get(model_name, model_name)


def case_key(model_name, prompt_size):
    return f"{canonical_model_name(model_name)}|{int(prompt_size)}"


def save_json_atomic(path, payload):
    """Write state without leaving a partially written JSON file after a crash."""
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as state_file:
        json.dump(payload, state_file, indent=2)
        state_file.write("\n")
        state_file.flush()
        os.fsync(state_file.fileno())
    os.replace(temp_path, path)


def load_benchmark_state(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as state_file:
        state = json.load(state_file)
    if not isinstance(state, dict) or not isinstance(state.get("cases"), list):
        raise ValueError("state file does not contain a valid cases list")
    return state


def cleanup_temp_artifacts():
    targets = {STATE_FILE, STATE_FILE + ".tmp"}
    for directory in {SCRIPT_DIR, os.getcwd()}:
        targets.update(glob.glob(os.path.join(directory, "temp_prompt_*.txt")))

    removed = []
    for path in sorted(targets):
        if not os.path.isfile(path):
            continue
        os.remove(path)
        removed.append(path)
    return removed


def build_initial_state(model_patterns, prompt_sizes):
    now = iso_now()
    cases = []
    for model_name in model_patterns:
        for prompt_size in prompt_sizes:
            cases.append({
                "key": case_key(model_name, prompt_size),
                "model": model_name,
                "prompt_size": int(prompt_size),
                "status": "pending",
                "attempts": 0,
                "last_start": None,
                "last_end": None,
                "model_path": None,
                "command": None,
                "error": None,
                "result": None,
            })
    return {
        "version": 1,
        "created_at": now,
        "updated_at": now,
        "models": list(model_patterns),
        "prompt_sizes": [int(size) for size in prompt_sizes],
        "micro_batch_size": int(MICRO_BATCH_SIZE),
        "cases": cases,
    }


def merge_benchmark_state(existing, fresh):
    """Reuse completed cases and make interrupted cases eligible for retry."""
    if not isinstance(existing, dict):
        return fresh, 0

    old_cases = existing.get("cases", [])
    old_by_key = {}
    for case in old_cases:
        if not isinstance(case, dict):
            continue
        try:
            migrated_key = case_key(case.get("model"), case.get("prompt_size"))
        except (TypeError, ValueError):
            continue
        old_by_key[migrated_key] = case
    recovered = 0
    preserved_fields = (
        "status",
        "attempts",
        "last_start",
        "last_end",
        "model_path",
        "command",
        "error",
        "result",
    )

    for case in fresh["cases"]:
        old_case = old_by_key.get(case["key"])
        if not old_case:
            continue
        for field in preserved_fields:
            if field in old_case:
                case[field] = old_case[field]
        if case.get("status") == "running":
            case["status"] = "pending"
            case["last_end"] = iso_now()
            case["error"] = "Interrupted during the previous run; queued for retry."
            recovered += 1

    fresh["created_at"] = existing.get("created_at", fresh["created_at"])
    fresh["updated_at"] = iso_now()
    return fresh, recovered


def results_from_state(state, model_patterns):
    results = {model_name: {} for model_name in model_patterns}
    for case in state.get("cases", []):
        model_name = case.get("model")
        result = case.get("result")
        if model_name in results and result is not None:
            results[model_name][int(case["prompt_size"])] = result
    return results


def main(model_patterns, prompt_sizes, reset_state=False, clean_temp=False):
    if clean_temp:
        removed = cleanup_temp_artifacts()
        print(f"Clean-temp mode complete. Removed {len(removed)} artifact(s).")
        for path in removed:
            print(f"  Removed: {path}")
        return

    if not os.path.exists(LLAMA_BIN):
        print(f"Error: llama-cli binary not found at {LLAMA_BIN}")
        print("Please run setup.sh first.")
        return

    fresh_state = build_initial_state(model_patterns, prompt_sizes)
    existing_state = None
    if not reset_state:
        try:
            existing_state = load_benchmark_state(STATE_FILE)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Error: could not load benchmark state from {STATE_FILE}: {exc}")
            print("Use --reset-state to intentionally replace the invalid state file.")
            return

    state, recovered_cases = merge_benchmark_state(existing_state, fresh_state)
    save_json_atomic(STATE_FILE, state)
    if reset_state:
        print(f"Started a fresh benchmark state at {STATE_FILE}")
    elif existing_state is not None:
        print(f"Restored benchmark state from {STATE_FILE}")
    else:
        print(f"Created benchmark state at {STATE_FILE}")
    if recovered_cases:
        print(f"Recovered {recovered_cases} interrupted case(s); they will be retried.")

    cases_by_key = {case["key"]: case for case in state["cases"]}
    results = results_from_state(state, model_patterns)

    for model_name, pattern in model_patterns.items():
        model_cases = [case for case in state["cases"] if case["model"] == model_name]
        if model_cases and all(case.get("status") == "done" for case in model_cases):
            print(f"All requested {model_name} cases are already cached; skipping model.")
            continue

        model_path = find_or_download_model(model_name, pattern)
        if not model_path:
            print(f"Warning: Model {model_name} not found and download failed/missing.")
            continue
        
        print(f"Benchmarking {model_name} ({model_path})...")
        
        # Run calibration only for cases that are not already cached.
        pending_prompt_sizes = [
            int(case["prompt_size"])
            for case in model_cases
            if case.get("status") != "done"
        ]
        run_calibration_test(model_path, pending_prompt_sizes)
        
        for size in prompt_sizes:
            case = cases_by_key[case_key(model_name, size)]
            if case.get("status") == "done":
                print(f"  Prompt size {size} already completed; using cached result.")
                continue

            print(f"  Running prompt size {size}...")
            case["status"] = "running"
            case["attempts"] = int(case.get("attempts", 0)) + 1
            case["last_start"] = iso_now()
            case["last_end"] = None
            case["model_path"] = model_path
            case["command"] = None
            case["error"] = None
            case["result"] = None
            state["updated_at"] = iso_now()
            save_json_atomic(STATE_FILE, state)

            case_result = None
            case_error = None
            case_status = "failed"
            temp_prompt_file = None
            master_fd = None
            slave_fd = None
            proc = None

            try:
                # Get initial prompt estimate, then calibrate to exact token count
                initial_prompt = get_prompt_text(size * 2)  # Get extra text for calibration room
                calibrated_prompt, actual_tokens = calibrate_token_count(model_path, initial_prompt, size)

                print(f"    Calibrated: target={size}, actual={actual_tokens} tokens")

                temp_prompt_file = f"temp_prompt_{size}.txt"
                with open(temp_prompt_file, 'w') as f:
                    f.write(calibrated_prompt)

                # Context size needs to be big enough.
                ctx = actual_tokens + 128
                batch_size = actual_tokens
                if MICRO_BATCH_SIZE > 0:
                    ub = MICRO_BATCH_SIZE
                elif MICRO_BATCH_SIZE == -2:
                    ub = 4096 if actual_tokens > 4096 else max(1, actual_tokens // 2)
                else:
                    ub = actual_tokens

                # Construct command
                cmd = [
                    LLAMA_BIN,
                    "-m", model_path,
                    "-f", temp_prompt_file,
                    "-n", "32",
                    "-c", str(ctx),
                    "-b", str(batch_size),
                    "-ub", str(ub),
                    "--temp", "0",
                    "-ngl", "99",
                    "-no-cnv"
                ]
                case["command"] = cmd
                state["updated_at"] = iso_now()
                save_json_atomic(STATE_FILE, state)
                # Use PTY to ensure all output is captured (unbuffered/tty behavior)
                master_fd, slave_fd = pty.openpty()
                
                env = os.environ.copy()
                if HSA_TARGET_VERSION is not None:
                    env["HSA_OVERRIDE_GFX_VERSION"] = HSA_TARGET_VERSION
                
                proc = subprocess.Popen(
                    cmd, 
                    stdout=slave_fd, 
                    stderr=slave_fd, 
                    stdin=slave_fd, 
                    env=env,
                    preexec_fn=os.setsid
                    # text=False, bufsize=0 # default is binary
                )
                
                # Close slave in parent
                os.close(slave_fd)
                slave_fd = None
                
                # Monitor output from master_fd
                
                start_time = time.time()
                collected_output = ""
                sent_signal = False
                
                # Make non-blocking
                os.set_blocking(master_fd, False)
                
                while True:
                    if proc.poll() is not None:
                        # Process ended, read remaining
                        try:
                            # Read until empty
                            while True: 
                                chunk = os.read(master_fd, 4096)
                                if chunk:
                                    decoded = chunk.decode('utf-8', errors='ignore')
                                    collected_output += decoded
                                    print(decoded, end='', flush=True) 
                                else: break
                        except OSError: pass
                        break
                        
                    reads = [master_fd]
                    ret = select.select(reads, [], [], 1.0) # 1 sec timeout
                    
                    data_read = False
                    if ret[0]:
                        try:
                            chunk = os.read(master_fd, 4096)
                            if chunk:
                                decoded = chunk.decode('utf-8', errors='ignore')
                                collected_output += decoded
                                print(decoded, end='', flush=True)
                                data_read = True
                        except OSError: pass
                        
                    # Filter ANSI strictly (m and K)
                    clean_output = re.sub(r'\x1b\[.*?[@-~]', '', collected_output)
                    
                    # Detect common_perf_print output (the authoritative metrics)
                    detailed_match = re.search(r"common_perf_print:.*prompt eval time\s*=", clean_output)
                    
                    # Also check for simple format (used by some tools)
                    is_simple_metrics = ("Prompt:" in clean_output and "Generation:" in clean_output)
                    
                    # Check if generation is complete (waiting for input prompt ">")
                    waiting_for_input = ">" in clean_output[-50:] and not detailed_match
                    
                    if waiting_for_input and not sent_signal:
                        # Send Enter twice to trigger perf output (like Ctrl+D or EOF)
                        print("\n  Generation complete, sending Enter for perf stats...")
                        try:
                            os.write(master_fd, b"\n")
                            time.sleep(0.5)
                            os.write(master_fd, b"\n")
                        except OSError:
                            pass
                        sent_signal = True  # Mark as handled, wait for perf output
                    
                    if (is_simple_metrics or detailed_match) and sent_signal:
                        print("\n  Metrics detected, stopping...")
                        proc.send_signal(signal.SIGINT)
                        
                        # Wait for exit
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        break

                    if time.time() - start_time > 180:
                        print("Timeout reached, killing process.")
                        proc.kill()
                        break
                
                # Parse metrics from accumulated output
                metrics = parse_results(collected_output)

                if metrics:
                    ttft = metrics.get('pp_ms', 0)
                    pp_tps = metrics.get('pp_tps', 0)
                    gen_tps = metrics.get('gen_tps', 0)
                    actual_tokens = metrics.get('pp_tokens', 0)
                    gen_tokens = metrics.get('gen_tokens', 0)
                    
                    # Ensure we have TTFT (ms)
                    if ttft == 0 and pp_tps > 0:
                        ttft = (size / pp_tps) * 1000
                        
                    # Calculate Gen Latency (ms/token)
                    if gen_tps > 0:
                        gen_ms_per_tok = 1000 / gen_tps
                    else:
                        gen_ms_per_tok = 0
                    
                    # Log token count comparison
                    print(f"    Requested tokens: {size}, Actual (llama-cli): {actual_tokens}")
                    if actual_tokens > 0 and abs(actual_tokens - size) > size * 0.1:
                        print(f"    WARNING: Token count differs by >10%!")
                    print(f"    TTFT: {ttft:.2f} ms")
                    print(f"    Gen Latency: {gen_ms_per_tok:.2f} ms/tok (generated {gen_tokens} tokens)")
                    
                    case_result = {
                        "ttft": ttft,
                        "gen_ms_tok": gen_ms_per_tok,
                        "requested_tokens": size,
                        "actual_tokens": actual_tokens,
                        "gen_tokens": gen_tokens
                    }
                    case_status = "done"
                else:
                    print("    Could not parse metrics.")
                    print(f"DEBUG: Collected Output:\n{collected_output[-500:]}")
                    case_result = "Parse Error"
                    case_error = "Could not parse benchmark metrics from llama.cpp output."

            except Exception as e:
                print(f"    Exception: {e}")
                case_result = "Error"
                case_error = str(e)
            finally:
                if slave_fd is not None:
                    try:
                        os.close(slave_fd)
                    except OSError:
                        pass
                if master_fd is not None:
                    try:
                        os.close(master_fd)
                    except OSError:
                        pass
                if proc is not None and proc.poll() is None:
                    proc.kill()
                if temp_prompt_file and os.path.exists(temp_prompt_file):
                    os.remove(temp_prompt_file)

            results[model_name][size] = case_result
            case["status"] = case_status
            case["last_end"] = iso_now()
            case["error"] = case_error
            case["result"] = case_result
            state["updated_at"] = iso_now()
            save_json_atomic(STATE_FILE, state)
            print(f"  Saved {model_name} prompt size {size} to {STATE_FILE}")

            case_index = state["cases"].index(case)
            has_more_runs = any(
                later_case.get("status") != "done"
                for later_case in state["cases"][case_index + 1:]
            )
            if has_more_runs:
                print(f"  Cooling down for {COOLDOWN_SEC} seconds before next run...")
                time.sleep(COOLDOWN_SEC)

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY (TTFT ms / Gen ms/tok)")
    print("="*80)
    
    # We will format cells as "1234.5 / 123.4"
    
    model_column_width = max(15, *(len(model_name) for model_name in results))
    header = f"{'Model':<{model_column_width}} | " + " | ".join([f"{s:<15}" for s in prompt_sizes])
    print(header)
    print("-" * len(header))
    
    for model_name, data in results.items():
        row = f"{model_name:<{model_column_width}} | "
        for size in prompt_sizes:
            val = data.get(size, "N/A")
            if isinstance(val, dict):
                cell = f"{val['ttft']:.1f} / {val['gen_ms_tok']:.2f}"
                row += f"{cell:<15} | "
            else:
                row += f"{str(val):<15} | "
        print(row)

def positive_int(value):
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("prompt sizes must be positive integers")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark selected llama.cpp models.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_PATTERNS,
        default=DEFAULT_MODELS,
        metavar="MODEL",
        help=(
            f"models to benchmark (default: {' '.join(DEFAULT_MODELS)}; "
            f"choices: {', '.join(MODEL_PATTERNS)})"
        ),
    )
    parser.add_argument(
        "--prompt-sizes",
        nargs="+",
        type=positive_int,
        default=DEFAULT_PROMPT_SIZES,
        metavar="TOKENS",
        help=(
            "prompt token sizes to benchmark "
            f"(default: {' '.join(map(str, DEFAULT_PROMPT_SIZES))})"
        ),
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help=f"discard cached progress and start fresh (state file: {STATE_FILE})",
    )
    parser.add_argument(
        "--clean-temp",
        action="store_true",
        help="remove benchmark.json and leftover temporary prompt/state files, then exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_model_patterns = {
        model_name: MODEL_PATTERNS[model_name] for model_name in args.models
    }
    main(
        selected_model_patterns,
        args.prompt_sizes,
        reset_state=args.reset_state,
        clean_temp=args.clean_temp,
    )
