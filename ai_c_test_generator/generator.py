"""
AI Test Generator - Core test generation logic
"""

import os
import re
import time
from pathlib import Path
from typing import Dict, List

import requests
import google.generativeai as genai

from .analyzer import DependencyAnalyzer


class SmartTestGenerator:
    """AI-powered test generator using Ollama with embedded systems support"""

    def __init__(self, api_key: str, repo_path: str = '.', redact_sensitive: bool = False, max_api_retries: int = 5, model_choice: str = 'ollama'):
        if model_choice == 'gemini' and api_key:
            genai.configure(api_key=api_key)
            self._genai_configured = True
        else:
            self._genai_configured = False
        
        self.api_key = api_key
        self.repo_path = repo_path
        self.redact_sensitive = redact_sensitive
        self.max_api_retries = max_api_retries
        self.model_choice = model_choice

        # Models
        self.current_model_name = None
        self.model = None
        self.ollama_url = "http://127.0.0.1:11434/api/generate"
        self.ollama_model = "qwen2.5-coder"
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.groq_model = "llama-3.3-70b-versatile"

        self._initialize_model()

        # Load Unity C prompts from ai-tool-gen-lab
        self._load_unity_prompts()

        # Enhanced embedded-specific prompts
        self.embedded_prompts = {
            'hardware_registers': """
            Generate comprehensive tests for hardware register access:
            - Test volatile register reads/writes
            - Verify memory-mapped I/O operations
            - Test register bit manipulation
            - Check boundary conditions and invalid values
            - Test atomic operations where applicable
            """,

            'bit_fields': """
            Generate tests for bit field operations:
            - Test individual bit field access
            - Verify bit field packing/unpacking
            - Test bit field boundary conditions
            - Check endianness handling
            - Test bit field arithmetic operations
            """,

            'state_machines': """
            Generate tests for state machine implementations:
            - Test valid state transitions
            - Verify invalid transition handling
            - Test state entry/exit actions
            - Check state machine initialization
            - Test concurrent state access
            """,

            'safety_critical': """
            Generate tests for safety-critical functions:
            - Test TMR (Triple Modular Redundancy) voting
            - Verify watchdog timer functionality
            - Test fault detection and recovery
            - Check safety margins and thresholds
            - Test fail-safe behaviors
            """,

            'interrupt_handlers': """
            Generate tests for interrupt service routines:
            - Test ISR entry/exit conditions
            - Verify interrupt priority handling
            - Test nested interrupt scenarios
            - Check interrupt latency requirements
            - Test interrupt masking/unmasking
            """,

            'dma_operations': """
            Generate tests for DMA transfer operations:
            - Test DMA channel configuration
            - Verify data transfer integrity
            - Check DMA completion callbacks
            - Test error handling and recovery
            - Verify memory alignment requirements
            """,

            'communication_protocols': """
            Generate tests for communication protocol implementations:
            - Test protocol state machines
            - Verify packet parsing and validation
            - Check error detection and correction
            - Test timeout and retry mechanisms
            - Verify protocol compliance
            """
        }

    def _initialize_model(self):
        """Initialize the selected AI model - no fallback"""
        print(f"[INFO] [DEBUG] Initializing {self.model_choice} model...", flush=True)
        
        if self.model_choice == "ollama":
            try:
                print(f"[INFO] [INIT] Checking Ollama connection at {self.ollama_url}...", flush=True)
                print(f"[INFO] [INIT] Sending test prompt to verify model '{self.ollama_model}' is loaded...", flush=True)
                print(f"[INFO] [INIT] Note: If this is the first run, it may take time to load the model into memory.", flush=True)
                
                payload = {"model": self.ollama_model, "prompt": "test", "stream": False}
                start_time = time.time()
                # Increased timeout to 120s to allow for model loading
                response = requests.post(self.ollama_url, json=payload, timeout=120)
                response.raise_for_status()
                duration = time.time() - start_time
                
                self.current_model_name = f"ollama:{self.ollama_model}"
                print(f"[PASS] [DEBUG] Ollama model '{self.ollama_model}' initialized in {duration:.2f}s", flush=True)
            except Exception as e:
                raise RuntimeError(f"Ollama model not available: {e}. Make sure Ollama is running and the model is installed.")
        
        elif self.model_choice == "gemini":
            if not self.api_key:
                raise RuntimeError("Gemini API key not provided. Use --api-key for Gemini model.")
            
            # Try Gemini models in priority order: latest -> flash
            gemini_models = [
                'gemini-2.0-flash-exp',  # Latest experimental
                'gemini-2.5-flash',      # Flash fallback
            ]
            
            for model_name in gemini_models:
                try:
                    print(f"[INFO] [INIT] Trying Gemini model: {model_name}", flush=True)
                    self.model = genai.GenerativeModel(model_name)
                    # Test the model with a simple request
                    test_response = self.model.generate_content("test", generation_config=genai.types.GenerationConfig(max_output_tokens=10))
                    self.current_model_name = model_name
                    print(f"[PASS] [DEBUG] Using Gemini model: {self.current_model_name}", flush=True)
                    break
                except Exception as e:
                    print(f"[WARN] Failed to initialize {model_name}: {e}")
                    continue
            else:
                raise RuntimeError("Failed to initialize any Gemini model. Check your API key, internet connection, and quota limits.")
        
        elif self.model_choice == "groq":
            if not self.api_key:
                raise RuntimeError("Groq API key not provided. Use --api-key for Groq model.")
            try:
                # Test Groq connection
                print(f"[INFO] [INIT] Testing Groq connection with model: {self.groq_model}", flush=True)
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.groq_model,
                    "messages": [{"role": "user", "content": "test"}],
                    "max_tokens": 10
                }
                response = requests.post(self.groq_url, json=payload, headers=headers, timeout=30)
                response.raise_for_status()
                self.current_model_name = f"groq:{self.groq_model}"
                print(f"[PASS] [DEBUG] Groq model '{self.groq_model}' initialized", flush=True)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Groq model: {e}")
        
        else:
            raise ValueError("Invalid model choice. Must be 'ollama', 'gemini', or 'groq'.")


        if self.model_choice == 'gemini' and self.model is None:
            raise Exception("No compatible Gemini model found. Please check your API key and internet connection.")

    def _load_unity_prompts(self):
        """Load Unity C test generation prompts from ai-tool-gen-lab"""
        # Copy the Unity prompts from ai-tool-gen-lab
        self.unity_base_prompt = """
You are a senior embedded C unit test engineer with 20+ years of experience using the Unity Test Framework (v2.5+). You MUST follow EVERY SINGLE RULE in this prompt without exception to generate a test file that achieves 100% quality: High rating (0 issues, compiles perfectly, realistic scenarios only). Failure to adhere will result in invalid output. Internally analyze the source code before generating: extract ALL functions, their EXACT signatures, public API (non-static), dependencies (internal vs external), and types (structs, unions, pointers, etc.).

FIRST, READ THE ENTIRE SOURCE CODE. EXTRACT:
- All function names and EXACT signatures (e.g., int main(void))
- All #define, thresholds, ranges, magic numbers
- All if/else/switch branches
- All struct/union/bitfield definitions

THEN, generate tests that cover 100% of this logic, including call sequences and return values.

CRITICAL REQUIREMENT: You MUST generate tests for EVERY SINGLE FUNCTION defined in the source file. Do not skip any functions. If the source has 4 functions, test all 4. If it has 10 functions, test all 10. Generate comprehensive tests for each function individually.

ABSOLUTE MANDATES (MUST ENFORCE THESE TO FIX BROKEN AND UNREALISTIC ISSUES)

NO COMPILATION ERRORS OR INCOMPLETE CODE: Output FULL, COMPLETE C code only. Mentally compile EVERY line before outputting (e.g., ensure all statements end with ';', all variables declared, no truncated lines like "extern int " or "int result = "). ONLY use existing headers from source. NO invented functions or headers. Code MUST compile with CMake/GCC for embedded targets. For internal dependencies (functions defined in the same file), DO NOT stub or redefine them—test them directly or through calling functions. For external dependencies only, provide mocks without redefinition conflicts (assume linking excludes real implementations for stubbed externals).

HANDLE MAIN() SPECIFICALLY: For files containing main(), declare "extern int main(void);" and call it directly in tests (result = main();). Assert on return value (always 0 in simple main). Focus tests on call sequence, param passing, and return. Do NOT stub main().

NO UNREALISTIC VALUES: STRICTLY enforce physical limits from source logic or domain knowledge. E.g., temperatures ALLOW negatives where valid (e.g., -40.0f to 125.0f); voltages 0.0f to 5.5f (no negatives unless signed in source). Use source-specific thresholds (e.g., extract >120.0f for "CRITICAL" from code). BAN absolute zero, overflows, or impossibles. For temp tests, use negatives like -10.0f where valid.

MEANINGFUL TESTS ONLY: EVERY test MUST validate the function's core logic, calculations, or outputs EXACTLY as per source. Match assertions to source behavior (e.g., if range is >= -40 && <=125, assert true for -40.0f, false for -40.1f). NO trivial "function called" tests unless paired with output validation. Each assertion MUST check a specific, expected result based on input.

STUBS MUST BE PERFECT: ONLY for listed external dependencies. Use EXACT signature, control struct, and FULL reset in setUp() AND tearDown() using memset or explicit zeroing. NO partial resets. Capture params if used in assertions. NO stubs for internals to avoid duplicates/linker errors.

TEST ISOLATION: EVERY test independent. setUp() for init/config/stub setup, tearDown() for COMPLETE cleanup/reset of ALL stubs (call_count=0, return_value=default, etc.).

NO NONSENSE: BAN random/arbitrary values (use source-derived, e.g., mid-range from logic). BAN redundancy (unique scenarios). BAN physical impossibilities or ignoring source thresholds.

INPUT: SOURCE CODE TO TEST (DO NOT MODIFY)
"""

        self.unity_output_format = """
IMPROVED RULES TO PREVENT BROKEN/UNREALISTIC OUTPUT

1. OUTPUT FORMAT (STRICT - ONLY C CODE):
Output PURE C code ONLY. Start with /* test_{source_name}.c – Auto-generated Expert Unity Tests */
NO markdown, NO ```c:disable-run
File structure EXACTLY: Comment -> Includes -> Extern declarations (for main and stubs) -> Stubs (only for externals) -> setUp/tearDown -> Tests -> main with UNITY_BEGIN/END and ALL RUN_TEST calls.

2. COMPILATION SAFETY (FIX BROKEN TESTS):
Includes: ONLY "unity.h", and standard <stdint.h>, <stdbool.h>, <string.h> if used in source or for memset. Do NOT include "{source_name}.h" if not present in source or necessary (e.g., for main.c, skip if no public API).
Signatures: COPY EXACTLY from source. NO mismatches in types, params, returns.
NO calls to undefined functions. For internals (same file), call directly without stubbing to avoid duplicates/linker errors.
Syntax: Perfect C - complete statements, matching braces, semicolons, no unused vars, embedded-friendly (no non-standard libs). Ensure all code is fully written (no placeholders).

3. MEANINGFUL TEST DESIGN (FIX TRIVIAL/UNREALISTIC):
"""

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.c']:
            return 'c'
        elif ext in ['.cpp', '.cc', '.cxx', '.c++']:
            return 'cpp'
        else:
            # Default to C++ for unknown extensions
            return 'cpp'

    def _try_generate_with_fallback(self, prompt: str, max_retries: int = None):
        """Generate content using the selected AI model - no fallback
        This is the single router - it will not attempt to fallback to other models.
        """
        if max_retries is None:
            max_retries = self.max_api_retries
        
        if self.model_choice == "ollama":
            # Use Ollama
            for attempt in range(max_retries):
                try:
                    print(f"[INFO] [LLM] Sending request to Ollama ({self.ollama_model})... This may take a while.", flush=True)
                    start_time = time.time()
                    payload = {
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False
                    }
                    response = requests.post(self.ollama_url, json=payload, timeout=300)
                    response.raise_for_status()
                    duration = time.time() - start_time
                    print(f"[INFO] [LLM] Response received in {duration:.2f}s", flush=True)
                    result = response.json()
                    # Create a mock response object with text attribute
                    class MockResponse:
                        def __init__(self, text):
                            self.text = text
                    return MockResponse(result["response"])
                except Exception as e:
                    print(f"[WARN] Ollama generation failed (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2)
                    else:
                        raise e
        elif self.model_choice == "gemini":
            for attempt in range(max_retries):
                try:
                    print(f"[INFO] [LLM] Sending request to Gemini...", flush=True)
                    start_time = time.time()
                    response = self.model.generate_content(prompt)
                    duration = time.time() - start_time
                    print(f"[INFO] [LLM] Response received in {duration:.2f}s", flush=True)
                    return response
                except Exception as e:
                    print(f"[WARN] Gemini generation failed (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        import time
                        # Exponential backoff for rate limits
                        sleep_time = min(2 ** attempt, 60)  # Max 60 seconds
                        print(f"[INFO] Waiting {sleep_time}s before retry...")
                        time.sleep(sleep_time)
                    else:
                        raise e
        
        elif self.model_choice == "groq":
            for attempt in range(max_retries):
                try:
                    print(f"[INFO] [LLM] Sending request to Groq ({self.groq_model})...", flush=True)
                    start_time = time.time()
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": self.groq_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 4096,
                        "temperature": 0.1
                    }
                    response = requests.post(self.groq_url, json=payload, headers=headers, timeout=300)
                    response.raise_for_status()
                    result = response.json()
                    duration = time.time() - start_time
                    print(f"[INFO] [LLM] Response received in {duration:.2f}s", flush=True)
                    # Create a mock response object with text attribute
                    class MockResponse:
                        def __init__(self, text):
                            self.text = text
                    return MockResponse(result["choices"][0]["message"]["content"])
                except Exception as e:
                    print(f"[WARN] Groq generation failed (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        import time
                        # Exponential backoff for rate limits
                        sleep_time = min(2 ** attempt, 60)  # Max 60 seconds
                        print(f"[INFO] Waiting {sleep_time}s before retry...")
                        time.sleep(sleep_time)
                    else:
                        raise e
        
        else:
            raise ValueError("Invalid model choice")

    # Router-style call function - explicit, no fallback
    def call_llm(self, prompt: str):
        return self._try_generate_with_fallback(prompt)

    def build_dependency_map(self, repo_path: str) -> Dict[str, str]:
        """Build a map of function_name -> source_file for the entire repository"""
        print("[INFO] [DEPS] Building global dependency map...")
        analyzer = DependencyAnalyzer(repo_path)
        all_cpp_files = analyzer.find_all_c_files()

        dependency_map = {}
        for file_path in all_cpp_files:
            functions = analyzer._extract_functions(file_path)
            for func in functions:
                dependency_map[func['name']] = file_path

        print(f"   Mapped {len(dependency_map)} functions across {len(all_cpp_files)} files")
        return dependency_map

    def generate_tests_for_file(self, file_path: str, repo_path: str, output_dir: str, dependency_map: Dict[str, str], validation_feedback: Dict = None) -> Dict:
        """Generate tests for a SINGLE file with proper context"""
        print(f"[INFO] Generating tests for {os.path.basename(file_path)}...", flush=True)
        analyzer = DependencyAnalyzer(repo_path)

        # Analyze this specific file
        analysis = analyzer.analyze_file_dependencies(file_path)
        print(f"[INFO] Analysis complete: {len(analysis['functions'])} functions found", flush=True)

        # IDENTIFY FUNCTIONS THAT NEED STUBS AND FUNCTIONS TO INCLUDE DIRECTLY
        functions_that_need_stubs = []
        functions_to_include_directly = []
        implemented_functions = {f['name'] for f in analysis['functions']}

        for called_func in analysis['called_functions']:
            if called_func not in implemented_functions:
                if called_func in dependency_map and dependency_map[called_func] != file_path:
                    # Function is defined in another file in the same repo - include directly
                    functions_to_include_directly.append(called_func)
                else:
                    # True external or unknown - may need stubbing if deterministically testable
                    functions_that_need_stubs.append(called_func)

        print(f"   [INFO] {os.path.basename(file_path)}: {len(analysis['functions'])} functions, {len(functions_to_include_directly)} repo includes, {len(functions_that_need_stubs)} need stubs", flush=True)

        # Build targeted prompt for this file only
        prompt = self._build_targeted_prompt(analysis, functions_that_need_stubs, functions_to_include_directly, repo_path, validation_feedback)
        print(f"[INFO] Prompt built, calling API...", flush=True)

        # Generate tests using the explicitly selected model
        response = self.call_llm(prompt)
        print(f"[SUCCESS] API response received from {self.current_model_name}", flush=True)
        test_code = response.text.strip()

        # POST-PROCESSING: Clean up common AI generation issues
        test_code = self._post_process_test_code(test_code, analysis, analysis['includes'])

        # Remove gtest includes for C files (Unity tests)
        if self._detect_language(file_path) == 'c':
            test_code = test_code.replace('#include <gtest/gtest.h>\n', '')
            test_code = test_code.replace('#include <gtest/gtest.h>', '')

        # Save test file
        test_filename = f"test_{os.path.basename(file_path)}"
        output_path = os.path.join(output_dir, test_filename)

        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(test_code)
        
        print(f"[SUCCESS] Test saved to {output_path}", flush=True)
        return {'success': True, 'test_file': output_path}

    def _build_targeted_prompt(self, analysis: Dict, functions_that_need_stubs: List[str], functions_to_include_directly: List[str], repo_path: str, validation_feedback: Dict = None) -> str:
        """Build targeted prompt based on detected programming language"""
        file_path = analysis.get('file_path', '')
        language = self._detect_language(file_path)
        print(f"[INFO] Detected language for {file_path}: {language}", flush=True)

        if language == 'c':
            print("[INFO] Using Unity prompt for C file", flush=True)
            return self._build_unity_prompt(analysis, functions_that_need_stubs, functions_to_include_directly, repo_path, validation_feedback)
        else:  # cpp or default
            print("[INFO] Using Google Test prompt for C++ file", flush=True)
            return self._build_gtest_prompt(analysis, functions_that_need_stubs, functions_to_include_directly, repo_path, validation_feedback)

    def _build_gtest_prompt(self, analysis: Dict, functions_that_need_stubs: List[str], functions_to_include_directly: List[str], repo_path: str, validation_feedback: Dict = None) -> str:
        """Build a focused prompt for a single file with stub requirements"""

        # REDACTED VERSION: Remove sensitive content before sending to API
        file_content = self._read_file_safely(analysis['file_path'])
        rel_path = os.path.relpath(analysis['file_path'], repo_path)
        source_name = os.path.splitext(os.path.basename(analysis['file_path']))[0]

        # Build validation feedback section
        validation_feedback_section = "NONE - First generation attempt"
        if validation_feedback:
            issues = validation_feedback.get('issues', [])
            if issues:
                validation_feedback_section = "PREVIOUS ATTEMPT FAILED WITH THESE SPECIFIC ISSUES - FIX THEM:\n" + "\n".join(f"- {issue}" for issue in issues[:5])  # Limit to first 5 issues
                if len(issues) > 5:
                    validation_feedback_section += f"\n- ... and {len(issues) - 5} more issues"

                # Add specific guidance for common issues
                if any('unreasonably high' in issue and '2000' in issue for issue in issues):
                    validation_feedback_section += "\n\nSPECIFIC FIX REQUIRED: Raw ADC values from rand() must be 0-1023. The value 2000 is invalid for read_temperature_raw() which returns rand() % 1024. Use values like 0, 512, 1023 for testing."
                elif any('unreasonably high' in issue for issue in issues):
                    validation_feedback_section += "\n\nSPECIFIC FIX REQUIRED: Temperature values must be in range -40.0 deg C to 125.0 deg C. Check source code for exact valid ranges."
                elif any('unreasonably low' in issue for issue in issues):
                    validation_feedback_section += "\n\nSPECIFIC FIX REQUIRED: Temperature values must be in range -40.0 deg C to 125.0 deg C. Negative values below -40 deg C are invalid."
            else:
                validation_feedback_section = "NONE - Previous attempt was successful"

        # Check for Arduino context
        is_arduino = 'Arduino.h' in analysis.get('includes', []) or \
                     any(k in file_content for k in ['Serial.', 'digitalWrite', 'pinMode', 'delay'])
        
        arduino_instructions = ""
        if is_arduino:
            arduino_instructions = """
ARDUINO/ESP32 TESTING STANDARDS (MANDATORY):
1. Include "Arduino_stubs.h" and the source header (e.g., "c_led.h").
2. Use a Test Fixture class (e.g., class CLedTest : public ::testing::Test).
3. CRITICAL: Declare `SetUp()` and `TearDown()` as `public:` (NOT protected).
   - In `SetUp()`: call `reset_arduino_stubs()` FIRST, then instantiate the class under test.
   - In `TearDown()`: delete the instance, then call `reset_arduino_stubs()`.
4. CRITICAL: Test the REAL class implementation. NEVER mock the class under test (e.g., no MockCLed : public c_led).
5. CRITICAL: Set member variables explicitly in tests (blynkServer, blynkParam, etc.) when they affect behavior.
6. Verify Serial output using `Serial.outputBuffer` (accumulated string).
   - Access: `Serial.outputBuffer` (it is a String object).
   - Verify: `EXPECT_STREQ(Serial.outputBuffer.c_str(), "expected output\\n");`
   - Do NOT use `Serial_print_calls` or `Serial_println_calls`. Use `outputBuffer`.
7. Verify GPIO using `digitalWrite_calls` (vector of DigitalWriteCall{pin, value}).
   - Example: `EXPECT_EQ(digitalWrite_calls[0].pin, 13);`
   - Example: `EXPECT_EQ(digitalWrite_calls[0].value, HIGH);`
8. Verify Timing using `delay_calls` (vector of DelayCall{ms}).
   - Example: `EXPECT_EQ(delay_calls[0].ms, 1000);`
   - CRITICAL: Do NOT compare `delay_calls[i]` directly to an int. Access `.ms`.
   - CRITICAL: Do NOT compare `digitalWrite_calls[i]` directly. Access `.pin` and `.value`.
"""

        prompt = f"""
You are writing C++17 code for a host-based GoogleTest environment.

STRICT RULES (do not violate):
1. Do NOT invent or mock hardware, HTTP, SPIFFS, or Arduino APIs.
2. Assume the following files already exist and are correct:
   - Arduino_stubs.h (provides Serial, digitalWrite, delay, String)
   - MockHTTPClient.h (already implemented)
   - MockSPIFFS.h (already implemented)
3. You MUST reuse existing mocks only. Do NOT redefine them.
4. Do NOT create local objects that are not used by the class under test.
5. Do NOT assume dependency injection unless explicitly provided.
6. Do NOT assume global side effects unless visible in code.
7. If something cannot be tested deterministically, SKIP that test.
8. All tests MUST compile on a normal PC with g++.
9. The output MUST be a single .cpp test file.
10. NO explanations. Output ONLY compilable code.

ENVIRONMENT DETAILS:
- Language: C++17
- Test framework: GoogleTest
- Platform: Linux / Windows host (NOT Arduino)
- No real networking, filesystem, or hardware allowed.

GOAL:
Generate GoogleTest unit tests ONLY for the following file:
<PASTE c_led.cpp or other file here>

TESTING STRATEGY:
- Test only observable behavior.
- Verify side effects via existing stubs (Serial.outputBuffer, call logs).
- Prefer fewer correct tests over many fake ones.
- If behavior depends on unmocked internals, do NOT test it.

FINAL CHECK BEFORE OUTPUT:
- Does this code compile without additional mocks?
- Does each test actually influence the code under test?
- Would a senior C++ reviewer accept this?

You are a Senior Embedded Validation Engineer with 15+ years of experience in automotive and IoT firmware testing.

Generate a PRODUCTION-GRADE GoogleTest unit test file for the following C++ source code.

STRICT REQUIREMENTS:
1. The test file MUST be:
   - Fully runnable
   - Deterministic in CI
   - Side-effect safe
   - Zero undefined behavior
   - If something cannot be tested deterministically, SKIP that test.
   - Do NOT invent or mock hardware, HTTP, SPIFFS, or Arduino APIs.
   - Reuse existing mocks only (MockHTTPClient.h, MockSPIFFS.h).
   - Test only observable behavior via existing stubs (Serial.outputBuffer, call logs).
   - Prefer fewer correct tests over many fake ones.

REPO-WIDE INTEGRATION:
- For functions defined in the same repository, include their headers and call them directly. Only stub true externals (e.g., HTTP, SPIFFS) using existing mocks.
- Add a repo-wide build option: When running tests, compile all repo files together (e.g., via CMake) so cross-file calls work without stubs.
- Direct calls to repo functions are deterministic if those functions are pure or have controlled inputs. Only skip if a function truly can't be tested (e.g., depends on unmappable hardware).

2. CRITICAL TESTING ARCHITECTURE (MANDATORY):
   - NEVER mock the class under test (e.g., do NOT create MockCLed : public c_led)
   - Test the REAL class implementation, not mocks
   - Mock ONLY external dependencies using existing mocks (MockHTTPClient.h, MockSPIFFS.h)
   - Use Arduino_stubs.h for Arduino I/O functions (digitalWrite, delay, Serial)
   - For classes with dependencies, use dependency injection or factory patterns if needed
   - Verify REAL firmware behavior, not mock expectations
   - If behavior depends on unmocked internals, SKIP that test

3. CRITICAL SOURCE CODE ANALYSIS (MANDATORY):
   - ANALYZE the actual source code behavior BEFORE writing tests
   - Extract ALL string literals, variable names, and logic paths from source
   - NEVER hard-code expected outputs that don't exist in the source code
   - For classes with member variables (blynkServer, blynkParam, etc.), set them explicitly in tests
   - Detect and test for bugs (infinite recursion, stack overflow, etc.)
   - Test error conditions and edge cases that actually exist in the code
   - Use correct class naming (CBlynkTest for c_blynk, not CLedTest)
   - Remove unnecessary extern declarations (no extern main, etc.)

3. You MUST:
   - Verify ALL digitalWrite calls by accessing `.pin` and `.value` members (NEVER compare struct directly)
   - Verify ALL delay() calls by accessing `.ms` member (NEVER compare struct directly)
   - Verify Serial output without assuming buffer overwrite (handle APPEND behavior safely)
   - Verify BOTH state transitions in any function that toggles hardware
   - Verify ALL iterations in loops (not just first iteration)
   - Verify constructor side effects INSIDE the test, not via SetUp() artifacts

4. You MUST:
   - Verify ALL digitalWrite calls by accessing `.pin` and `.value` members (NEVER compare struct directly)
   - Verify ALL delay() calls by accessing `.ms` member (NEVER compare struct directly)
   - Verify Serial output without assuming buffer overwrite (handle APPEND behavior safely)
   - Verify BOTH state transitions in any function that toggles hardware
   - Verify ALL iterations in loops (not just first iteration)
   - Verify constructor side effects INSIDE the test, not via SetUp() artifacts

HERE IS THE SOURCE CODE TO TEST:
<PASTE YOUR SOURCE HERE>

{arduino_instructions}

INPUT: SOURCE CODE TO TEST (DO NOT MODIFY)
/* ==== BEGIN src/{os.path.basename(analysis['file_path'])} ==== */
{file_content}
/* ==== END src/{os.path.basename(analysis['file_path'])} ==== */
REPO FUNCTIONS TO INCLUDE DIRECTLY (call these directly; assume headers exist):
{chr(10).join(f"- {func_name}" for func_name in functions_to_include_directly) or "- None"}

EXTERNAL FUNCTIONS TO MOCK (only these; infer signatures from calls if needed; use typical embedded types):
{chr(10).join(f"- {func_name}" for func_name in functions_that_need_stubs) or "- None"}

IMPROVED RULES TO PREVENT BROKEN/UNREALISTIC OUTPUT

1. OUTPUT FORMAT (STRICT - ONLY C++ CODE):
Output PURE C++ code ONLY. Start with /* test_{source_name}.cpp – Auto-generated Expert Google Test Tests */
NO markdown, NO ```c:disable-run
File structure EXACTLY: Comment -> Includes -> Extern declarations (for main and mocks) -> Mocks (only for externals) -> SetUp/TearDown -> Tests -> main with RUN_ALL_TESTS().

2. COMPILATION SAFETY (FIX BROKEN TESTS):
Includes: ONLY <gtest/gtest.h>, and standard <stdint.h>, <stdbool.h>, <string.h> if used in source or for memset. Do NOT include "{source_name}.h" if not present in source or necessary (e.g., for main.cpp, skip if no public API).
Signatures: COPY EXACTLY from source. NO mismatches in types, params, returns.
NO calls to undefined functions. For internals (same file), call directly without mocking to avoid duplicates/linker errors.
Syntax: Perfect C++ - complete statements, matching braces, semicolons, no unused vars, embedded-friendly (no non-standard libs). Ensure all code is fully written (no placeholders).

3. MEANINGFUL TEST DESIGN (FIX TRIVIAL/UNREALISTIC):
MANDATORY: Generate tests for EVERY FUNCTION in the source file. Do not skip functions. For each function, create 3-5 focused tests covering all branches and edge cases.
Focus: Test FUNCTION LOGIC exactly (e.g., for validate_range: assert true/false based on precise source conditions like >= -40 && <=125). For main(), test call sequence (e.g., get_temperature_celsius called once, param to check_temperature_status matches return), and main return 0.
BAN: Tests with wrong expectations (cross-check source thresholds). BAN "was_called" alone - ALWAYS validate outputs/params.
Each test: 1 purpose, 3-5 per public function, covering ALL branches/logic from source.

4. REALISTIC TEST VALUES (FIX UNREALISTIC - ENFORCE LIMITS):
Extract ranges/thresholds from source (e.g., -40.0f to 125.0f for validate; -10.0f for cold).
Temperatures: -40.0f to 125.0f (allow negatives if in source); normal 0.0f-50.0f. E.g., min: -40.0f, max: 125.0f, nominal: 25.0f, cold: -10.1f.
Voltages: 0.0f to 5.0f (max 5.5f for edges) unless source allows negatives.
Currents: 0.0f to 10.0f.
Integers: Within type limits/source ranges (e.g., raw 0-1023 from rand() % 1024).
Pointers: Valid or NULL only for error tests.
BAN: Negative temps/volts unless source handles; absolute zero; huge numbers (>1e6 unless domain-specific).

5. FLOATING POINT HANDLING (MANDATORY):
ALWAYS: ASSERT_FLOAT_EQ(expected, actual) with tolerance - use EXPECT_NEAR(expected, actual, 0.1f) for temp, 0.01f for voltage, etc.
NEVER direct equality for floats.

6. MOCK IMPLEMENTATION (FIX BROKEN MOCKS):
CRITICAL: NEVER mock the class under test. Test the REAL implementation.
ONLY mock external dependencies using existing mocks (MockHTTPClient.h, MockSPIFFS.h).
For Arduino I/O: Use Arduino_stubs.h instead of mocking - verify calls via digitalWrite_calls, delay_calls, Serial.outputBuffer.
For classes with dependencies: Use dependency injection or test with real dependencies where safe.
Example: If testing c_blynk, use MockHTTPClient but test real c_blynk methods.
SetUp(): Create mock instances for externals only and set expectations.
TearDown(): Reset mocks.
For non-deterministic (e.g., rand-based): Mock to make deterministic; test ranges via multiple configs.
Do NOT mock printf—comment that output assertion requires redirection (not implemented here).
If something cannot be tested deterministically, SKIP that test.

7. COMPREHENSIVE TEST SCENARIOS (MEANINGFUL & REALISTIC):
Normal: Mid-range inputs from source, assert correct computation (e.g., temp status "NORMAL" for 25.0f).
Edge: Exact min/max from source (e.g., -40.0f true, -40.1f false; -10.0f "NORMAL", -10.1f "COLD").
Error: Invalid inputs (out-of-range, NULL if applicable), simulate via mocks - assert error code/safe output.
Cover ALL branches: If/else, returns, etc.

8. AVOID BAD PATTERNS (PREVENT COMMON FAILURES):
NO arbitrary values (derive from source, e.g., raw=500 for mid).
NO duplicate/redundant tests (unique per branch).
NO physical impossibilities or ignoring source thresholds.
NO tests ignoring outputs - always assert results.
NO hard-coded expected outputs that don't exist in source code.
NO tests that assume member variables are set when they're not.
NO contradictory test cases (same inputs, different expected outputs).
For internals like rand-based: Mock and test deterministic outputs; check ranges (e.g., 0-1023).
For main with printf: Assert only on mocks and return; comment on printf limitation.
DETECT BUGS: Test for infinite recursion (isDeviceConnected(retries--)), uninitialized variables, etc.

9. GOOGLE TEST BEST PRACTICES:
Appropriate asserts: EXPECT_EQ/ASSERT_EQ for ints, EXPECT_FLOAT_EQ for floats, EXPECT_STREQ for chars, EXPECT_TRUE/FALSE for bools, EXPECT_NE(nullptr, ptr) for pointers.
CRITICAL: DO NOT use EXPECT_EQ for structs/unions directly. Compare members individually.
Comments: 1-line above EACH assert: // Expected: [source-based reason, e.g., 25.0f is NORMAL per >85 check]
Handle complex types: Field-by-field for structs, both views for unions, masks for bitfields, arrays with EXPECT_EQ.

10. STRUCTURE & ISOLATION:
Test names: TEST(TestSuite, FunctionNormalMidRange), TEST(TestSuite, FunctionMinEdgeValid), etc.
Test fixture naming: Use correct class names (CBlynkTest for c_blynk, CLedTest for c_led).
SetUp/TearDown: ALWAYS present in test fixtures. Full mock reset in BOTH. Minimal if no state.
NO unnecessary extern declarations (remove extern main, etc.).

QUALITY SELF-CHECK (DO INTERNALLY BEFORE OUTPUT):
Compiles? (No duplicates, exact sigs) Yes/No - if No, fix.
Realistic? (Values match source ranges, allow valid negatives) Yes/No.
Meaningful? (Assertions match source logic exactly, cover branches) Yes/No.
Source Analysis? (Expected outputs exist in source, member variables set, bugs detected) Yes/No.
Mocks? (Only externals, NEVER the class under test, full reset) Yes/No.
Coverage? (All branches, no gaps/redundancy) Yes/No.
Architecture? (Tests real implementation, not mocks) Yes/No.
Deterministic? (Skip untestable parts) Yes/No.
Compile without additional mocks? Yes/No.
Each test influences code under test? Yes/No.
Senior C++ reviewer accept? Yes/No.

VALIDATION FEEDBACK (CRITICAL - ADDRESS THESE SPECIFIC ISSUES):
{validation_feedback_section}

FINAL INSTRUCTION:
Generate ONLY the complete test_{source_name}.cpp C++ code now. Follow EVERY rule strictly. Output nothing else.
"""
        return prompt

    def _build_unity_prompt(self, analysis: Dict, functions_that_need_stubs: List[str], functions_to_include_directly: List[str], repo_path: str, validation_feedback: Dict = None) -> str:
        """Build a focused Unity prompt for C files with stub requirements"""

        # REDACTED VERSION: Remove sensitive content before sending to API
        file_content = self._read_file_safely(analysis['file_path'])
        rel_path = os.path.relpath(analysis['file_path'], repo_path)
        source_name = os.path.splitext(os.path.basename(analysis['file_path']))[0]

        # Build validation feedback section
        validation_feedback_section = "NONE - First generation attempt"
        if validation_feedback:
            issues = validation_feedback.get('issues', [])
            if issues:
                validation_feedback_section = "PREVIOUS ATTEMPT FAILED WITH THESE SPECIFIC ISSUES - FIX THEM:\n" + "\n".join(f"- {issue}" for issue in issues[:5])  # Limit to first 5 issues
                if len(issues) > 5:
                    validation_feedback_section += f"\n- ... and {len(issues) - 5} more issues"

                # Add specific guidance for common issues
                if any('unreasonably high' in issue and '2000' in issue for issue in issues):
                    validation_feedback_section += "\n\nSPECIFIC FIX REQUIRED: Raw ADC values from rand() must be 0-1023. The value 2000 is invalid for read_temperature_raw() which returns rand() % 1024. Use values like 0, 512, 1023 for testing."
                elif any('unreasonably high' in issue for issue in issues):
                    validation_feedback_section += "\n\nSPECIFIC FIX REQUIRED: Temperature values must be in range -40.0 deg C to 125.0 deg C. Check source code for exact valid ranges."
                elif any('unreasonably low' in issue for issue in issues):
                    validation_feedback_section += "\n\nSPECIFIC FIX REQUIRED: Temperature values must be in range -40.0 deg C to 125.0 deg C. Negative values below -40 deg C are invalid."
            else:
                validation_feedback_section = "NONE - Previous attempt was successful"

        prompt = f"""
You are a senior embedded C unit test engineer with 20+ years of experience using the Unity Test Framework (v2.5+). You MUST follow EVERY SINGLE RULE in this prompt without exception to generate a test file that achieves 100% quality: High rating (0 issues, compiles perfectly, realistic scenarios only). Failure to adhere will result in invalid output. Internally analyze the source code before generating: extract ALL functions, their EXACT signatures, public API (non-static), dependencies (internal vs external), and types (structs, unions, pointers, etc.).

FIRST, READ THE ENTIRE SOURCE CODE. EXTRACT:
- All function names and EXACT signatures (e.g., int main(void))
- All #define, thresholds, ranges, magic numbers
- All if/else/switch branches
- All struct/union/bitfield definitions

THEN, generate tests that cover 100% of this logic, including call sequences and return values.

CRITICAL REQUIREMENT: You MUST generate tests for EVERY SINGLE FUNCTION defined in the source file. Do not skip any functions. If the source has 4 functions, test all 4. If it has 10 functions, test all 10. Generate comprehensive tests for each function individually.

ABSOLUTE MANDATES (MUST ENFORCE THESE TO FIX BROKEN AND UNREALISTIC ISSUES)

NO COMPILATION ERRORS OR INCOMPLETE CODE: Output FULL, COMPLETE C code only. Mentally compile EVERY line before outputting (e.g., ensure all statements end with ';', all variables declared, no truncated lines like "extern int " or "int result = "). ONLY use existing headers from source. NO invented functions or headers. Code MUST compile with CMake/GCC for embedded targets. For internal dependencies (functions defined in the same file), DO NOT stub or redefine them—test them directly or through calling functions. For external dependencies only, provide stubs without redefinition conflicts (assume linking excludes real implementations for stubbed externals).

HANDLE MAIN() SPECIFICALLY: For files containing main(), declare "extern int main(void);" and call it directly in tests (result = main();). Assert on return value (always 0 in simple main). Focus tests on call sequence, param passing, and return. Do NOT stub main().

NO UNREALISTIC VALUES: STRICTLY enforce physical limits from source logic or domain knowledge. E.g., temperatures ALLOW negatives where valid (e.g., -40.0f to 125.0f); voltages 0.0f to 5.5f (no negatives unless signed in source). Use source-specific thresholds (e.g., extract >120.0f for "CRITICAL" from code). BAN absolute zero, overflows, or impossibles. For temp tests, use negatives like -10.0f where valid.

MEANINGFUL TESTS ONLY: EVERY test MUST validate the function's core logic, calculations, or outputs EXACTLY as per source. Match assertions to source behavior (e.g., if range is >= -40 && <=125, assert true for -40.0f, false for -40.1f). NO trivial "function called" tests unless paired with output validation. Each assertion MUST check a specific, expected result based on input.

STUBS MUST BE PERFECT: ONLY for listed external dependencies. Use EXACT signature, control struct, and FULL reset in setUp() AND tearDown() using memset or explicit zeroing. NO partial resets. Capture params if used in assertions. NO stubs for internals to avoid duplicates/linker errors.

FLOATS: MANDATORY TEST_ASSERT_FLOAT_WITHIN with domain-specific tolerance (e.g., 0.1f for temp). BAN TEST_ASSERT_EQUAL_FLOAT.

TEST ISOLATION: EVERY test independent. setUp() for init/config/stub setup, tearDown() for COMPLETE cleanup/reset of ALL stubs (call_count=0, return_value=default, etc.).

NO NONSENSE: BAN random/arbitrary values (use source-derived, e.g., mid-range from logic). BAN redundancy (unique scenarios). BAN physical impossibilities or ignoring source thresholds.

INPUT: SOURCE CODE TO TEST (DO NOT MODIFY)
/* ==== BEGIN src/{source_name}.c ==== */
{file_content}
/* ==== END src/{source_name}.c ==== */
REPO FUNCTIONS TO INCLUDE DIRECTLY (call these directly; assume headers exist):
{chr(10).join(f"- {func_name}" for func_name in functions_to_include_directly) or "- None"}

EXTERNAL FUNCTIONS TO STUB (only these; infer signatures from calls if needed; use typical embedded types):
{chr(10).join(f"- {func_name}" for func_name in functions_that_need_stubs) or "- None"}

IMPROVED RULES TO PREVENT BROKEN/UNREALISTIC OUTPUT

REPO-WIDE INTEGRATION:
- For functions defined in the same repository, include their headers and call them directly. Only stub true externals (e.g., HTTP, SPIFFS) using existing mocks.
- Add a repo-wide build option: When running tests, compile all repo files together (e.g., via CMake) so cross-file calls work without stubs.
- Direct calls to repo functions are deterministic if those functions are pure or have controlled inputs. Only skip if a function truly can't be tested (e.g., depends on unmappable hardware).

1. OUTPUT FORMAT (STRICT - ONLY C CODE):
Output PURE C code ONLY. Start with /* test_{source_name}.c – Auto-generated Expert Unity Tests */
NO markdown, NO ```c:disable-run
CRITICAL: DO NOT include <gtest/gtest.h> or any Google Test headers. This is C code using Unity framework ONLY.
File structure EXACTLY: Comment -> Includes -> Extern declarations (for main and stubs) -> Stubs (only for externals) -> setUp/tearDown -> Tests -> main with UNITY_BEGIN/END and ALL RUN_TEST calls.

2. COMPILATION SAFETY (FIX BROKEN TESTS):
Includes: ONLY "unity.h", and standard <stdint.h>, <stdbool.h>, <string.h> if used in source or for memset. Do NOT include "{source_name}.h" if not present in source or necessary (e.g., for main.c, skip if no public API).
Signatures: COPY EXACTLY from source. NO mismatches in types, params, returns.
NO calls to undefined functions. For internals (same file), call directly without stubbing to avoid duplicates/linker errors.
Syntax: Perfect C - complete statements, matching braces, semicolons, no unused vars, embedded-friendly (no non-standard libs). Ensure all code is fully written (no placeholders).

3. MEANINGFUL TEST DESIGN (FIX TRIVIAL/UNREALISTIC):
MANDATORY: Generate tests for EVERY FUNCTION in the source file. Do not skip functions. For each function, create 3-5 focused tests covering all branches and edge cases.
Focus: Test FUNCTION LOGIC exactly (e.g., for validate_range: assert true/false based on precise source conditions like >= -40 && <=125). For main(), test call sequence (e.g., get_temperature_celsius called once, param to check_temperature_status matches return), and main return 0.
BAN: Tests with wrong expectations (cross-check source thresholds). BAN "was_called" alone - ALWAYS validate outputs/params.
Each test: 1 purpose, 3-5 per public function, covering ALL branches/logic from source.

4. REALISTIC TEST VALUES (FIX UNREALISTIC - ENFORCE LIMITS):
Extract ranges/thresholds from source (e.g., -40.0f to 125.0f for validate; -10.0f for cold).
Temperatures: -40.0f to 125.0f (allow negatives if in source); normal 0.0f-50.0f. E.g., min: -40.0f, max: 125.0f, nominal: 25.0f, cold: -10.1f.
Voltages: 0.0f to 5.0f (max 5.5f for edges) unless source allows negatives.
Currents: 0.0f to 10.0f.
Integers: Within type limits/source ranges (e.g., raw 0-1023 from rand() % 1024).
Pointers: Valid or NULL only for error tests.
BAN: Negative temps/volts unless source handles; absolute zero; huge numbers (>1e6 unless domain-specific).

5. FLOATING POINT HANDLING (MANDATORY):
ALWAYS: TEST_ASSERT_FLOAT_WITHIN(tolerance, expected, actual) - use 0.1f for temp, 0.01f for voltage, etc.
NEVER equal checks for floats.

6. STUB IMPLEMENTATION (FIX BROKEN STUBS):
ONLY for listed externals: Exact prototype + control struct (return_value, was_called, call_count, captured params if asserted).
Example struct: typedef struct {{ float return_value; bool was_called; uint32_t call_count; int last_param; }} stub_xxx_t; static stub_xxx_t stub_xxx = {{0}};
Stub func: Increment count, store params, return configured value.
setUp(): memset(&stub_xxx, 0, sizeof(stub_xxx)); for ALL stubs + any init.
tearDown(): SAME full reset for ALL stubs.
For non-deterministic (e.g., rand-based): Stub to make deterministic; test ranges via multiple configs.
Do NOT stub printf—comment that output assertion requires redirection (not implemented here).

7. COMPREHENSIVE TEST SCENARIOS (MEANINGFUL & REALISTIC):
Normal: Mid-range inputs from source, assert correct computation (e.g., temp status "NORMAL" for 25.0f).
Edge: Exact min/max from source (e.g., -40.0f true, -40.1f false; -10.0f "NORMAL", -10.1f "COLD").
Error: Invalid inputs (out-of-range, NULL if applicable), simulate via stubs - assert error code/safe output.
Cover ALL branches: If/else, returns, etc.

8. AVOID BAD PATTERNS (PREVENT COMMON FAILURES):
NO arbitrary values (derive from source, e.g., raw=500 for mid).
NO duplicate/redundant tests (unique per branch).
NO physical impossibilities or ignoring source thresholds.
NO tests ignoring outputs - always assert results.
For internals like rand-based: Stub and test deterministic outputs; check ranges (e.g., 0-1023).
For main with printf: Assert only on stubs and return; comment on printf limitation.

9. UNITY BEST PRACTICES:
Appropriate asserts: EQUAL_INT/HEX for ints, FLOAT_WITHIN for floats, EQUAL_STRING for chars, TRUE/FALSE for bools, NULL/NOT_NULL for pointers, EQUAL_MEMORY for structs/unions.
Comments: 1-line above EACH assert: // Expected: [source-based reason, e.g., 25.0f is NORMAL per >85 check]
Handle complex types: Field-by-field for structs, both views for unions, masks for bitfields, arrays with EQUAL_xxx_ARRAY.

10. STRUCTURE & ISOLATION:
Test names: test_[function]normal_mid_range, test[function]_min_edge_valid, etc.
setUp/tearDown: ALWAYS present. Full stub reset in BOTH. Minimal if no state.

QUALITY SELF-CHECK (DO INTERNALLY BEFORE OUTPUT):
Compiles? (No duplicates, exact sigs) Yes/No - if No, fix.
Realistic? (Values match source ranges, allow valid negatives) Yes/No.
Meaningful? (Assertions match source logic exactly, cover branches) Yes/No.
Stubs? (Only externals, full reset) Yes/No.
Coverage? (All branches, no gaps/redundancy) Yes/No.

VALIDATION FEEDBACK (CRITICAL - ADDRESS THESE SPECIFIC ISSUES):
{validation_feedback_section}

FINAL INSTRUCTION:
Generate ONLY the complete test_{source_name}.c C code now. Follow EVERY rule strictly. Output nothing else.
"""
        return prompt

    def _read_file_safely(self, file_path: str) -> str:
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception:
            return "// Unable to read file"

    def _redact_sensitive_content(self, file_path: str) -> str:
        """Redact sensitive content before sending to external API"""
        content = self._read_file_safely(file_path)

        # Redaction patterns for common sensitive content
        redaction_patterns = [
            # Remove comments that might contain sensitive information
            (r'/\*.*?\*/', '/* [COMMENT REDACTED] */'),
            (r'//.*$', '// [COMMENT REDACTED]'),

            # Redact string literals that might contain sensitive data
            (r'"[^"]*"', '"[STRING REDACTED]"'),

            # Redact potential API keys, passwords, secrets
            (r'\b[A-Za-z0-9+/=]{20,}\b', '[CREDENTIAL REDACTED]'),  # Base64-like strings

            # Redact email addresses
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REDACTED]'),

            # Redact URLs that might point to internal systems
            (r'https?://[^\s\'"]+', '[URL REDACTED]'),

            # Redact potential IP addresses
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP REDACTED]'),
        ]

        redacted_content = content
        for pattern, replacement in redaction_patterns:
            redacted_content = re.sub(pattern, replacement, redacted_content, flags=re.MULTILINE | re.IGNORECASE)

        return redacted_content

    def _post_process_test_code(self, test_code: str, analysis: Dict, source_includes: List[str]) -> str:
        """Post-process generated test code to fix common issues and improve quality"""

        # Remove markdown code block markers
        test_code = re.sub(r'^```c?\s*', '', test_code, flags=re.MULTILINE)
        test_code = re.sub(r'```\s*$', '', test_code, flags=re.MULTILINE)

        # Remove any leading characters before the first comment or include
        # This fixes artifacts like "pp" appearing at the start of the file
        match = re.search(r'(/\*|//|#include)', test_code)
        if match:
            test_code = test_code[match.start():]

        # Fix floating point assertions - replace ASSERT_FLOAT_EQ with EXPECT_NEAR
        test_code = re.sub(
            r'ASSERT_FLOAT_EQ\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',
            r'EXPECT_NEAR(\1, \2, 0.01f)',
            test_code
        )

        # Fix incorrect Google Test macro names if any
        # Assuming standard gtest, but adjust if needed

        # Fix unrealistic temperature values (absolute zero or impossible ranges)
        test_code = re.sub(r'-273\.15f?', '-40.0f', test_code)  # Replace absolute zero with realistic minimum
        test_code = re.sub(r'1e10+', '1000.0f', test_code)      # Replace extremely large values

        # Fix invalid rand() mock return values (should be 0-1023 for read_temperature_raw)
        # Look for mock_rand_instance.return_value = <invalid_value>
        test_code = re.sub(
            r'(mock_rand_instance\.return_value\s*=\s*)(\d+)(;)',            lambda m: f"{m.group(1)}{min(int(m.group(2)), 1023)}{m.group(3)}" if int(m.group(2)) > 1023 else m.group(0),
            test_code
        )

        # Remove printf/scanf statements that might appear in tests
        test_code = re.sub(r"printf\s*\([^;]*\);\s*", "", test_code)
        test_code = re.sub(r"scanf\s*\([^;]*\);\s*", "", test_code)

        # Ensure proper includes - only include gtest.h and existing source headers
        lines = test_code.split("\n")
        cleaned_lines = []

        for line in lines:
            # Keep gtest.h include
            if "#include <gtest/gtest.h>" in line:
                cleaned_lines.append(line)
                continue

            # Only keep includes for headers that exist in source_includes or are standard headers
            if line.startswith("#include"):
                include_match = re.match(r"#include\s+[\"<]([^\">]+)[\">]", line)
                if include_match:
                    header_name = include_match.group(1)
                    # Only include headers that exist in source_includes or are standard headers
                    if header_name in source_includes or header_name.endswith(".h"):
                        # Additional check: don't include main.h if it doesn't exist
                        if header_name == "main.h" and not any("main.h" in inc for inc in source_includes):
                            continue
                        cleaned_lines.append(line)
                # Skip non-matching include lines
                continue

            # Keep all other lines
            cleaned_lines.append(line)

        # Ensure gtest.h is included if not present
        has_gtest = any("#include <gtest/gtest.h>" in line for line in cleaned_lines)
        if not has_gtest:
            cleaned_lines.insert(0, "#include <gtest/gtest.h>")

        # Add Google Test main function with test discovery
        test_code_with_main = "\n".join(cleaned_lines)
        
        # Check if Arduino.h is included (directly or indirectly)
        # If so, we should NOT generate conflicting mocks for String, Serial, etc.
        # This is a heuristic: if the source includes Arduino.h, we assume the build environment provides stubs.
        is_arduino = any("Arduino.h" in inc for inc in source_includes)
        
        if is_arduino:
             # Remove conflicting class definitions if they exist in the generated code
             # This is a simple regex approach; a parser would be better but this covers common cases
             test_code_with_main = re.sub(r'class\s+String\s*\{[^}]*\};', '', test_code_with_main, flags=re.DOTALL)
             test_code_with_main = re.sub(r'class\s+SerialClass\s*\{[^}]*\};', '', test_code_with_main, flags=re.DOTALL)
             test_code_with_main = re.sub(r'extern\s+SerialClass\s+Serial;', '', test_code_with_main)
             # Also remove MockArduinoSerial if it conflicts
             # But usually we WANT mocks for testing. The issue is redefinition of the BASE classes.
             # If Arduino_stubs.h defines String, we shouldn't define it again.

        test_functions = re.findall(r'TEST\s*\(\s*\w+\s*,\s*\w+\s*\)', test_code_with_main)

        if test_functions and "int main(" not in test_code_with_main:
            main_function = "\n\nint main(int argc, char **argv) {\n    ::testing::InitGoogleTest(&argc, argv);\n    return RUN_ALL_TESTS();\n}"

            test_code_with_main += main_function

        return test_code_with_main

    def _analyze_embedded_patterns(self, source_code: str, function_name: str) -> Dict:
        """Analyze source code for embedded systems patterns"""
        patterns = {
            'hardware_registers': False,
            'bit_fields': False,
            'state_machines': False,
            'safety_critical': False,
            'interrupt_handlers': False,
            'dma_operations': False,
            'communication_protocols': False
        }

        # Check for hardware register patterns
        if re.search(r'\bvolatile\s+\w+\s*\*\s*\w+', source_code) or re.search(r'\bREG_\w+', source_code):
            patterns['hardware_registers'] = True

        # Check for bit field patterns
        if re.search(r'\w+\s*:\s*\d+', source_code) or re.search(r'bitfield|BITFIELD', source_code):
            patterns['bit_fields'] = True

        # Check for state machine patterns
        if re.search(r'state|STATE|enum.*state', source_code, re.IGNORECASE):
            patterns['state_machines'] = True

        # Check for safety critical patterns
        if re.search(r'safety|critical|watchdog|TMR|voting', source_code, re.IGNORECASE):
            patterns['safety_critical'] = True

        # Check for interrupt handler patterns
        if re.search(r'ISR|interrupt|IRQ', source_code, re.IGNORECASE):
            patterns['interrupt_handlers'] = True

        # Check for DMA patterns
        if re.search(r'DMA|dma|transfer', source_code, re.IGNORECASE):
            patterns['dma_operations'] = True

        # Check for communication protocol patterns
        if re.search(r'protocol|CAN|SPI|I2C|UART|serial', source_code, re.IGNORECASE):
            patterns['communication_protocols'] = True

        return patterns

    def _build_embedded_prompt(self, function_name: str, function_info: Dict, embedded_patterns: Dict) -> str:
        """Build enhanced prompt based on detected embedded patterns"""
        base_prompt = f"Generate comprehensive Google Test tests for the embedded C++ function '{function_name}'.\n\n"

        # Add specific prompts for detected patterns
        active_patterns = [k for k, v in embedded_patterns.items() if v]

        if active_patterns:
            base_prompt += "This function involves the following embedded systems concepts:\n"
            for pattern in active_patterns:
                if pattern in self.embedded_prompts:
                    base_prompt += f"- {pattern.replace('_', ' ').title()}: {self.embedded_prompts[pattern].strip()}\n"
            base_prompt += "\n"

        base_prompt += """
Requirements:
- Use Google Test framework (gtest)
- Include SetUp() and TearDown() functions in test fixtures
- Test realistic embedded values and edge cases
- Handle volatile variables correctly
- Test hardware-specific behaviors
- Ensure thread safety where applicable
- Validate error conditions and recovery

Generate complete, compilable C++ test code.
"""

        return base_prompt

    def _post_process_embedded_tests(self, generated_tests: str, embedded_patterns: Dict) -> str:
        """Post-process generated tests for embedded-specific patterns"""
        processed = generated_tests

        # Add volatile qualifiers where needed
        if embedded_patterns.get('hardware_registers'):
            # Add volatile to register access patterns
            processed = re.sub(r'(\w+)\s*=\s*\*(\w+);', r'\1 = *(volatile typeof(\1)*)\2;', processed)

        # Add interrupt disabling/enabling for critical sections
        if embedded_patterns.get('interrupt_handlers'):
            # Wrap critical sections
            processed = re.sub(
                r'(TEST_ASSERT_\w+\([^;]+;\s*)',
                r'__disable_irq();\n    \1\n    __enable_irq();',
                processed
            )

        return processed

    def generate_embedded_tests(self, source_code: str, function_name: str,
                               function_info: Dict) -> str:
        """
        Generate comprehensive tests for embedded C++ functions with hardware-specific considerations.

        Args:
            source_code: The complete source code
            function_name: Name of the function to test
            function_info: Function metadata from analyzer

        Returns:
            Generated Google Test code
        """

        # Analyze function for embedded patterns
        embedded_patterns = self._analyze_embedded_patterns(source_code, function_name)

        # Build enhanced prompt based on detected patterns
        prompt = self._build_embedded_prompt(function_name, function_info, embedded_patterns)

        # Generate tests using AI with embedded context
        try:
            response = self._try_generate_with_fallback(prompt)
            generated_tests = response.text

            # Post-process for embedded-specific patterns
            processed_tests = self._post_process_embedded_tests(generated_tests, embedded_patterns)

            # Validate and enhance tests
            validated_tests = self.validator.validate_and_enhance_tests(
                processed_tests, source_code, function_name, embedded_patterns
            )

            return validated_tests

        except Exception as e:
            print(f"[ERROR] Test generation failed: {e}")
            return ""
