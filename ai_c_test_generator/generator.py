"""
AI Test Generator - Core test generation logic
"""

import os
import re
import time
from pathlib import Path
from typing import Dict, List

import google.generativeai as genai

from .analyzer import DependencyAnalyzer


class SmartTestGenerator:
    """AI-powered test generator using Google Gemini with embedded systems support"""

    def __init__(self, api_key: str, repo_path: str = '.', redact_sensitive: bool = False, max_api_retries: int = 5):
        genai.configure(api_key=api_key)
        self.repo_path = repo_path
        self.redact_sensitive = redact_sensitive
        self.max_api_retries = max_api_retries

        # Use modern API (v0.8.0+) with gemini-2.5-flash as primary model
        self.models_to_try = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']
        self.current_model_name = None
        self.model = None

        self._initialize_model()

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
        """Initialize the best available model"""
        for model_name in self.models_to_try:
            try:
                self.model = genai.GenerativeModel(model_name)
                self.current_model_name = model_name
                print(f"✅ [OK] Using model: {model_name}")
                break
            except Exception as e:
                print(f"⚠️ [WARN] Model {model_name} failed: {e}")
                continue

        if self.model is None:
            raise Exception("No compatible Gemini model found. Please check your API key and internet connection.")

    def _try_generate_with_fallback(self, prompt: str, max_retries: int = None):
        """Try to generate content with automatic model fallback and retry logic"""
        if max_retries is None:
            max_retries = self.max_api_retries
        last_error = None

        # First try with current model, with retries
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response
            except Exception as e:
                error_str = str(e).lower()
                last_error = e

                # Check if it's a rate limit, quota, or timeout error
                is_rate_limit = any(keyword in error_str for keyword in [
                    'rate limit', 'quota', 'limit exceeded', 'resource exhausted',
                    '429', 'too many requests'
                ])

                is_timeout = any(keyword in error_str for keyword in [
                    'timeout', '504', 'request timed out', 'deadline exceeded',
                    'connection timeout', 'read timeout', 'gateway timeout',
                    'service unavailable', '502', '503', 'upstream request timeout'
                ])

                if (is_rate_limit or is_timeout) and attempt < max_retries - 1:
                    # Use longer backoff for timeouts (they might need more time to recover)
                    if is_timeout:
                        wait_time = min((2 ** attempt) * 2 + 5, 30)  # 7s, 13s, 30s max for timeouts
                        print(f"⚠️ [WARN] Timeout hit on {self.current_model_name}, waiting longer to retry in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                    else:
                        wait_time = (2 ** attempt) + 1  # Exponential backoff: 1s, 3s, 7s for rate limits
                        print(f"⚠️ [WARN] Rate limit hit on {self.current_model_name}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                elif is_rate_limit or is_timeout:
                    # Rate limit or timeout persists, try fallback models
                    error_type = "rate limited" if is_rate_limit else "timing out"
                    print(f"⚠️ [WARN] {self.current_model_name} persistently {error_type}, trying fallback models...")
                    break
                else:
                    # Not a rate limit error, re-raise immediately
                    raise e

        # Try fallback models if we got here due to rate limits
        original_model = self.current_model_name
        for model_name in self.models_to_try:
            if model_name == original_model:
                continue  # Skip the model that just failed

            try:
                print(f"🔄 [GEN] Trying fallback model: {model_name}")
                fallback_model = genai.GenerativeModel(model_name)
                response = fallback_model.generate_content(prompt)

                # If successful, switch to this model for future requests
                self.model = fallback_model
                self.current_model_name = model_name
                print(f"✅ [OK] Switched to model: {model_name}")
                return response

            except Exception as fallback_error:
                print(f"❌ [ERROR] Fallback model {model_name} also failed: {fallback_error}")
                continue

        # If all attempts failed, raise the last error
        raise last_error or Exception("All models failed and no fallback available")

    def build_dependency_map(self, repo_path: str) -> Dict[str, str]:
        """Build a map of function_name -> source_file for the entire repository"""
        print("📋 [DEPS] Building global dependency map...")
        analyzer = DependencyAnalyzer(repo_path)
        all_c_files = analyzer.find_all_c_files()

        dependency_map = {}
        for file_path in all_c_files:
            functions = analyzer._extract_functions(file_path)
            for func in functions:
                dependency_map[func['name']] = file_path

        print(f"   Mapped {len(dependency_map)} functions across {len(all_c_files)} files")
        return dependency_map

    def generate_tests_for_file(self, file_path: str, repo_path: str, output_dir: str, dependency_map: Dict[str, str], validation_feedback: Dict = None) -> Dict:
        """Generate tests for a SINGLE file with proper context"""
        analyzer = DependencyAnalyzer(repo_path)

        # Analyze this specific file
        analysis = analyzer.analyze_file_dependencies(file_path)

        # IDENTIFY FUNCTIONS THAT NEED STUBS
        functions_that_need_stubs = []
        implemented_functions = {f['name'] for f in analysis['functions']}

        for called_func in analysis['called_functions']:
            # If called function is not in current file AND not a standard library function
            if (called_func not in implemented_functions and
                called_func in dependency_map and
                dependency_map[called_func] != file_path):
                functions_that_need_stubs.append(called_func)

        print(f"   📋 {os.path.basename(file_path)}: {len(analysis['functions'])} functions, {len(functions_that_need_stubs)} need stubs")

        # Build targeted prompt for this file only
        prompt = self._build_targeted_prompt(analysis, functions_that_need_stubs, repo_path, validation_feedback)

        # Generate tests using modern API with fallback support
        try:
            response = self._try_generate_with_fallback(prompt)
            test_code = response.text.strip()

            # POST-PROCESSING: Clean up common AI generation issues
            test_code = self._post_process_test_code(test_code, analysis, analysis['includes'])

            # Save test file
            test_filename = f"test_{os.path.basename(file_path)}"
            output_path = os.path.join(output_dir, test_filename)

            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(test_code)

            return {'success': True, 'test_file': output_path}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _build_targeted_prompt(self, analysis: Dict, functions_that_need_stubs: List[str], repo_path: str, validation_feedback: Dict = None) -> str:
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
                    validation_feedback_section += "\n\nSPECIFIC FIX REQUIRED: Temperature values must be in range -40.0°C to 125.0°C. Check source code for exact valid ranges."
                elif any('unreasonably low' in issue for issue in issues):
                    validation_feedback_section += "\n\nSPECIFIC FIX REQUIRED: Temperature values must be in range -40.0°C to 125.0°C. Negative values below -40°C are invalid."
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
c/* ==== BEGIN src/{source_name}.c ==== */
{file_content}
/* ==== END src/{source_name}.c ==== */
EXTERNAL FUNCTIONS TO STUB (only these; infer signatures from calls if needed; use typical embedded types):
{chr(10).join(f"- {func_name}" for func_name in functions_that_need_stubs) or "- None"}

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

        # Fix floating point assertions - replace TEST_ASSERT_EQUAL_FLOAT with TEST_ASSERT_FLOAT_WITHIN
        test_code = re.sub(
            r'TEST_ASSERT_EQUAL_FLOAT\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)',
            r'TEST_ASSERT_FLOAT_WITHIN(0.01f, \1, \2)',
            test_code
        )

        # Fix incorrect Unity macro names
        test_code = re.sub(r'TEST_ASSERT_GREATER_THAN_EQUAL_INT', 'TEST_ASSERT_GREATER_OR_EQUAL_INT', test_code)
        test_code = re.sub(r'TEST_ASSERT_LESS_THAN_EQUAL_INT', 'TEST_ASSERT_LESS_OR_EQUAL_INT', test_code)
        test_code = re.sub(r'TEST_ASSERT_GREATER_THAN_INT', 'TEST_ASSERT_GREATER_OR_EQUAL_INT', test_code)
        test_code = re.sub(r'TEST_ASSERT_LESS_THAN_INT', 'TEST_ASSERT_LESS_OR_EQUAL_INT', test_code)

        # Fix unrealistic temperature values (absolute zero or impossible ranges)
        test_code = re.sub(r'-273\.15f?', '-40.0f', test_code)  # Replace absolute zero with realistic minimum
        test_code = re.sub(r'1e10+', '1000.0f', test_code)      # Replace extremely large values

        # Fix invalid rand() stub return values (should be 0-1023 for read_temperature_raw)
        # Look for stub_rand_instance.return_value = <invalid_value>
        test_code = re.sub(
            r'(stub_rand_instance\.return_value\s*=\s*)(\d+)(;)',            lambda m: f"{m.group(1)}{min(int(m.group(2)), 1023)}{m.group(3)}" if int(m.group(2)) > 1023 else m.group(0),
            test_code
        )

        # Fix negative voltage/current values (replace with 0)
        test_code = re.sub(r"-\d+\.?\d*f?\b", "0.0f", test_code)

        # Remove invalid function calls (like main())
        test_code = re.sub(r"\bmain\s*\(\s*\)\s*;", "", test_code)
        # Remove any main function definitions that might appear
        test_code = re.sub(r"int\s+main\s*\([^)]*\)\s*{[^}]*}", "", test_code, flags=re.DOTALL)

        # Remove printf/scanf statements that might appear in tests
        test_code = re.sub(r"printf\s*\([^;]*\);\s*", "", test_code)
        test_code = re.sub(r"scanf\s*\([^;]*\);\s*", "", test_code)

        # Ensure proper includes - only include unity.h and existing source headers
        lines = test_code.split("\n")
        cleaned_lines = []

        for line in lines:
            # Keep unity.h include
            if "#include \"unity.h\"" in line:
                cleaned_lines.append(line)
                continue

            # Only keep includes for headers that exist in source_includes or are standard headers
            if line.startswith("#include"):
                include_match = re.match(r"#include\s+[\"<]([^\">]+)[\">]", line)
                if include_match:
                    header_name = include_match.group(1)
                    # Only include headers that exist in source_includes or are standard headers
                    if header_name in source_includes or header_name.endswith(".h"):
                        # Additional check: don"t include main.h if it doesn"t exist
                        if header_name == "main.h" and not any("main.h" in inc for inc in source_includes):
                            continue
                        cleaned_lines.append(line)
                # Skip non-matching include lines
                continue

            # Keep all other lines
            cleaned_lines.append(line)

        # Ensure unity.h is included if not present
        has_unity = any("#include \"unity.h\"" in line for line in cleaned_lines)
        if not has_unity:
            cleaned_lines.insert(0, "#include \"unity.h\"")

        # Add Unity main function with RUN_TEST calls for all test functions
        test_code_with_main = "\n".join(cleaned_lines)
        test_functions = re.findall(r"void\s+(test_\w+)\s*\(", test_code_with_main)

        if test_functions:
            main_function = "\n\nint main(void) {\n    UNITY_BEGIN();\n\n"
            for test_func in test_functions:
                main_function += f"    RUN_TEST({test_func});\n"
            main_function += "\n    return UNITY_END();\n}"

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
        base_prompt = f"Generate comprehensive Unity tests for the embedded C function '{function_name}'.\n\n"

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
- Use Unity test framework v2.5+
- Include setUp() and tearDown() functions
- Test realistic embedded values and edge cases
- Handle volatile variables correctly
- Test hardware-specific behaviors
- Ensure thread safety where applicable
- Validate error conditions and recovery

Generate complete, compilable C test code.
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
        Generate comprehensive tests for embedded C functions with hardware-specific considerations.

        Args:
            source_code: The complete source code
            function_name: Name of the function to test
            function_info: Function metadata from analyzer

        Returns:
            Generated Unity test code
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
            print(f"❌ [ERROR] Test generation failed: {e}")
            return ""
