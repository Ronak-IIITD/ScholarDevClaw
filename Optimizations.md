# Repository Optimization Roadmap

This document outlines the technical debt, performance bottlenecks, and "barely working" implementations within the ScholarDevClaw codebase. It provides a clear guide for both human developers and AI agents to transition the system from a pure Python implementation to a high-performance, robust architecture.

## 🎯 Overview
The primary goal is to reduce latency in research intelligence, accelerate validation (fuzzing/mutation), and improve the reliability of the patch generation pipeline.

---

## 🚀 1. Computational & Vectorized Math (Python $\rightarrow$ NumPy/C++)
*Focus: Replacing manual loops with vectorized operations to achieve orders-of-magnitude speedups in mathematical computations.*

### 1.1 TF-IDF and Cosine Similarity
- **Target**: `core/src/scholardevclaw/research_intelligence/similarity.py` $\rightarrow$ `_tfidf_similarity`, `_compute_idf`
- **Current State**: Uses manual Python loops and `math.sqrt` to calculate dot products and vector magnitudes.
- **Proposed Change**: Replace manual loops with `numpy` array operations or `scikit-learn`'s `TfidfVectorizer` and `cosine_similarity`.
- **Why**: Vectorized operations in NumPy are implemented in C and operate on contiguous memory, bypassing Python's loop overhead.
- **Priority**: High

### 1.2 Fuzzer Input Generation
- **Target**: `core/src/scholardevclaw/validation/fuzzing.py` $\rightarrow$ `_generate_input`
- **Current State**: Uses generator expressions: `bytes(random.getrandbits(8) for _ in range(length))`.
- **Proposed Change**: Use `os.urandom(length)` for bytes and `random.choices(chars, k=length)` for strings.
- **Why**: These are optimized C-implemented functions in the standard library and are significantly faster than iterating in Python.
- **Priority**: Medium

### 1.3 Baseline Benchmarks
- **Target**: `core/src/scholardevclaw/validation/runner.py` $\rightarrow$ `_GENERIC_BENCH_SCRIPT`
- **Current State**: Uses nested `for` loops for matrix-like baseline computations.
- **Proposed Change**: Use `numpy` for any matrix operations used in benchmarking.
- **Why**: Ensures the baseline is fair and the benchmark infrastructure itself isn't the bottleneck.
- **Priority**: Low

---

## 🏗️ 2. Structural & Graph Traversal (Python $\rightarrow$ C++ Extension)
*Focus: Reducing the overhead of Python's object model during recursive traversal of massive trees and graphs.*

### 2.1 Call Graph Traversal
- **Target**: `core/src/scholardevclaw/repo_intelligence/call_graph.py` $\rightarrow$ `find_call_chain`, `find_all_callers`
- **Current State**: Implements BFS/DFS using Python `deque` and `set`.
- **Proposed Change**: Move the graph data structure and traversal logic into a C++ extension (using PyBind11).
- **Why**: For repositories with thousands of functions, the overhead of Python object creation and pointer chasing in `set`/`deque` becomes a major bottleneck.
- **Priority**: High

### 2.2 AST Walking
- **Target**: `core/src/scholardevclaw/repo_intelligence/tree_sitter_analyzer.py` $\rightarrow$ `_walk_for_elements`, `_walk_for_imports`
- **Current State**: Recursive Python traversal of nodes parsed by Tree-sitter.
- **Proposed Change**: Implement the walker logic in C++, interacting directly with the Tree-sitter C API.
- **Why**: Reduces the cost of crossing the Python-C boundary for every single node in the AST.
- **Priority**: Medium

### 2.3 PDF Text Extraction
- **Target**: `core/src/scholardevclaw/ingestion/pdf_parser.py` $\rightarrow$ `_extract_sections`, `_extract_equations`
- **Current State**: Heavy regex and string normalization inside nested loops over PDF spans.
- **Proposed Change**: Move the heuristic-based extraction and symbolic detection logic to C++.
- **Why**: PDF parsing involves high volumes of string analysis; C++ allows for more efficient memory management and faster regex execution.
- **Priority**: Medium

---

## ⚡ 3. Process & Execution Model (Python $\rightarrow$ Parallel/Async)
*Focus: Eliminating sequential bottlenecks and reducing the cost of process creation.*

### 3.1 Mutation Test Execution
- **Target**: `core/src/scholardevclaw/validation/mutation_testing.py` $\rightarrow$ `MutationTestRunner`
- **Current State**: Spawns a complete `pytest` subprocess via `subprocess.run` for every single mutation.
- **Proposed Change**: Implement a persistent test worker process or use an in-process execution model (e.g., `pytest` API).
- **Why**: Process startup time (Python interpreter load) dominates the execution time of the actual tests.
- **Priority**: Critical

### 3.2 Parallel Fuzzing
- **Target**: `core/src/scholardevclaw/validation/fuzzing.py` $\rightarrow$ `PythonFuzzer.fuzz`
- **Current State**: Fuzzing iterations (up to 10,000) run sequentially in a single thread.
- **Proposed Change**: Use `concurrent.futures.ProcessPoolExecutor` to distribute fuzzing iterations across all CPU cores.
- **Why**: Fuzzing is "embarrassingly parallel"; utilizing 8+ cores provides a near-linear speedup.
- **Priority**: High

### 3.3 Parallel Validation Runs
- **Target**: `core/src/scholardevclaw/validation/runner.py` $\rightarrow$ `ValidationRunner.run`
- **Current State**: Test and benchmark subprocesses are executed one-by-one.
- **Proposed Change**: Parallelize the execution of independent test suites.
- **Why**: Reduces total wall-clock time for validation cycles.
- **Priority**: Medium

### 3.4 Async LLM Generation
- **Target**: `core/src/scholardevclaw/patch_generation/generator.py` $\rightarrow$ `PatchGenerator`
- **Current State**: LLM synthesis calls for multiple files are performed sequentially.
- **Proposed Change**: Use `asyncio` and `aiohttp` (or an async SDK) to fire multiple LLM requests in parallel.
- **Why**: LLM API calls are I/O bound; waiting for one to finish before starting the next is a waste of time.
- **Priority**: High

---

## 🛠️ 4. Reliability & "Barely Working" Logic (Refactoring)
*Focus: Moving from fragile heuristics to robust, engineering-sound implementations.*

### 4.1 AST-Based Mutation
- **Target**: `core/src/scholardevclaw/validation/mutation_testing.py` $\rightarrow$ `mutate_file`
- **Current State**: Uses `content.replace(original, mutated, 1)`, which can replace the wrong occurrence of a string.
- **Proposed Change**: Perform mutations by modifying the AST (Abstract Syntax Tree) and then regenerating the code.
- **Why**: Guarantees the mutation is applied to the correct syntactic element.
- **Priority**: High

### 4.2 Robust Error Handling
- **Target**: `core/src/scholardevclaw/validation/runner.py` $\rightarrow$ `_run_bench_script`
- **Current State**: Uses broad `except Exception:` blocks.
- **Proposed Change**: Define specific custom exception classes and catch only the expected failure modes.
- **Why**: Broad exceptions mask critical bugs (e.g., MemoryErrors, KeyboardInterrupts) and make debugging significantly harder.
- **Priority**: Medium

### 4.3 Corpus Deduplication
- **Target**: `core/src/scholardevclaw/validation/fuzzing.py` $\rightarrow$ `FuzzerManager`
- **Current State**: Simple lists for seeds and crashes.
- **Proposed Change**: Implement a hashing-based deduplication system (e.g., using `hashlib` for content hashes) to track unique crashes.
- **Why**: Prevents the fuzzer from wasting cycles on redundant crash reports as the corpus grows.
- **Priority**: Low

---

## 🗺️ Execution Roadmap

### Phase 1: The Quick Wins (Immediate Impact)
- [ ] `similarity.py`: Vectorize math with NumPy.
- [ ] `fuzzing.py`: Optimize input generation.
- [ ] `generator.py`: Parallelize LLM requests with `asyncio`.

### Phase 2: Scaling the Infrastructure (Medium Effort)
- [ ] `fuzzing.py`: Implement `ProcessPoolExecutor` for parallel fuzzing.
- [ ] `mutation_testing.py`: Replace `subprocess.run` with a persistent worker.
- [ ] `runner.py`: Parallelize `pytest` execution.

### Phase 3: Deep Engineering (High Effort/Maximum Gain)
- [ ] `mutation_testing.py`: Transition to AST-level mutations.
- [ ] `call_graph.py`: Implement C++ graph traversal extension.
- [ ] `tree_sitter_analyzer.py`: Move AST walker to C++.
- [ ] `pdf_parser.py`: Move symbolic detection to C++.
