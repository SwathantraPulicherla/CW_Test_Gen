#!/usr/bin/env python3
"""
CLI interface for AI C Test Generator
"""

import argparse
import os
import sys
from pathlib import Path

# Add compatibility for older Python versions
try:
    from importlib.metadata import packages_distributions
except ImportError:
    # Python < 3.10 compatibility
    try:
        from importlib_metadata import packages_distributions
    except ImportError:
        # Fallback implementation
        def packages_distributions():
            return {}

from .generator import SmartTestGenerator
from .validator import TestValidator


def create_parser():
    """Create argument parser for the CLI tool"""
    parser = argparse.ArgumentParser(
        description="AI-powered C unit test generator using Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate tests for all C files in current directory
  ai-c-testgen --api-key YOUR_API_KEY

  # Generate tests for specific directory
  ai-c-testgen --repo-path /path/to/c/project --api-key YOUR_API_KEY

  # Use environment variable for API key
  export GEMINI_API_KEY=your_key_here
  ai-c-testgen --repo-path /path/to/c/project

  # Enable automatic regeneration for low-quality tests
  ai-c-testgen --repo-path /path/to/c/project --regenerate-on-low-quality --max-regeneration-attempts 3

  # Set quality threshold (only regenerate if below medium quality)
  ai-c-testgen --repo-path /path/to/c/project --regenerate-on-low-quality --quality-threshold medium
        """
    )

    parser.add_argument(
        '--repo-path',
        type=str,
        default='.',
        help='Path to the C repository (default: current directory)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='tests',
        help='Output directory for generated tests (default: tests)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        help='Google Gemini API key (can also use GEMINI_API_KEY env var)'
    )

    parser.add_argument(
        '--source-dir',
        type=str,
        default='src',
        help='Source directory containing C files (default: src)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    parser.add_argument(
        '--max-regeneration-attempts',
        type=int,
        default=2,
        help='Maximum number of regeneration attempts for low-quality tests (default: 2)'
    )

    parser.add_argument(
        '--regenerate-on-low-quality',
        action='store_true',
        help='Automatically regenerate tests that are validated as low quality'
    )

    parser.add_argument(
        '--redact-sensitive',
        action='store_true',
        help='Redact sensitive content (comments, strings, credentials) before sending to API'
    )

    parser.add_argument(
        '--quality-threshold',
        type=str,
        choices=['low', 'medium', 'high'],
        default='high',
        help='Quality threshold for regeneration (low, medium, high). Only regenerate tests below this threshold (default: high)'
    )

    parser.add_argument(
        '--max-api-retries',
        type=int,
        default=5,
        help='Maximum number of API retries for timeouts and rate limits (default: 5)'
    )

    return parser


def validate_environment(args):
    """Validate environment and arguments"""
    # Check repository path
    if not os.path.exists(args.repo_path):
        print(f"❌ Repository path '{args.repo_path}' does not exist")
        return False

    # Check for C files in source directory
    source_path = os.path.join(args.repo_path, args.source_dir)
    if not os.path.exists(source_path):
        print(f"❌ Source directory '{source_path}' does not exist")
        return False

    # Check for C files
    c_files = []
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.endswith(('.c', '.h')):
                c_files.append(os.path.join(root, file))

    if not c_files:
        print(f"❌ No C files found in '{source_path}'")
        return False

    # Check API key
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Set GEMINI_API_KEY environment variable or use --api-key")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        return False

    return True


def main():
    """Main CLI entry point"""
    print("🔄 [INFO] AI C Test Generator CLI started, initializing...")
    parser = create_parser()
    args = parser.parse_args()

    if not validate_environment(args):
        sys.exit(1)

    api_key = args.api_key or os.getenv('GEMINI_API_KEY')

    print("🚀 AI C Test Generator")
    print(f"   Repository: {args.repo_path}")
    print(f"   Source dir: {args.source_dir}")
    print(f"   Output dir: {args.output}")
    print()

    try:
        # Initialize components
        print("🔄 [INFO] Initializing Gemini model and validator...")
        generator = SmartTestGenerator(
            api_key,
            args.repo_path,
            redact_sensitive=args.redact_sensitive,
            max_api_retries=args.max_api_retries
        )
        validator = TestValidator(args.repo_path)

        # Build dependency map
        if args.verbose:
            print("📋 Building dependency map...")
        dependency_map = generator.build_dependency_map(args.repo_path)

        # Find C files in source directory (excluding main.c)
        source_path = os.path.join(args.repo_path, args.source_dir)
        c_files = []
        for root, dirs, files in os.walk(source_path):
            for file in files:
                if file.endswith('.c'):  # Only process .c files, not headers
                    # Skip main.c as it's not suitable for unit testing
                    if file == 'main.c':
                        if args.verbose:
                            print(f"⏭️ Skipping main.c (application entry point)")
                        continue
                    c_files.append(os.path.join(root, file))

        if args.verbose:
            print(f"📁 Found {len(c_files)} C files to process")

        # Create output directory
        output_dir = os.path.join(args.repo_path, args.output)
        os.makedirs(output_dir, exist_ok=True)

        # Clean up old compilation reports
        compilation_report_dir = os.path.join(output_dir, "compilation_report")
        if os.path.exists(compilation_report_dir):
            print("🧹 Cleaning up old compilation reports...")
            try:
                import shutil
                shutil.rmtree(compilation_report_dir)
            except (OSError, PermissionError) as e:
                print(f"⚠️ Could not clean up old reports: {e}")
                # Try to remove files individually
                try:
                    for root, dirs, files in os.walk(compilation_report_dir, topdown=False):
                        for file in files:
                            try:
                                os.remove(os.path.join(root, file))
                            except OSError:
                                pass
                        for dir_name in dirs:
                            try:
                                os.rmdir(os.path.join(root, dir_name))
                            except OSError:
                                pass
                    os.rmdir(compilation_report_dir)
                except OSError:
                    print("⚠️ Skipping cleanup due to permission issues")
        os.makedirs(compilation_report_dir, exist_ok=True)

        # Process each file
        successful_generations = 0
        validation_reports = []
        regeneration_stats = {'total_regenerations': 0, 'successful_regenerations': 0}

        for file_path in c_files:
            rel_path = os.path.relpath(file_path, args.repo_path)
            print(f"🎯 Processing: {rel_path}")

            max_attempts = args.max_regeneration_attempts + 1  # +1 for initial generation
            attempt = 0
            successful_results = []  # Keep all successful results
            final_validation = None

            while attempt < max_attempts:
                attempt += 1
                try:
                    # Generate tests for this file
                    result = generator.generate_tests_for_file(
                        file_path, args.repo_path, output_dir, dependency_map, final_validation if attempt > 1 else None
                    )

                    if not result['success']:
                        print(f"   ❌ Generation failed: {result['error']}")
                        break

                    # Validate the generated test
                    if args.verbose:
                        print(f"   🔍 Validating (attempt {attempt})...")
                    validation_result = validator.validate_test_file(result['test_file'], file_path)

                    # Store successful result (keep all successful attempts)
                    successful_results.append((result, validation_result))
                    final_validation = validation_result

                    # Check if regeneration is needed based on quality threshold
                    quality_levels = {'low': 0, 'medium': 1, 'high': 2}
                    current_quality_level = quality_levels.get(validation_result['quality'].lower(), 0)
                    threshold_quality_level = quality_levels.get(args.quality_threshold.lower(), 0)

                    needs_regeneration = (
                        args.regenerate_on_low_quality and
                        current_quality_level < threshold_quality_level and
                        attempt < max_attempts
                    )

                    # Print validation summary
                    status = "✅" if validation_result['compiles'] and validation_result['realistic'] else "⚠️"
                    quality = validation_result['quality']
                    compiles = 'Compiles' if validation_result['compiles'] else 'Broken'
                    realistic = 'Realistic' if validation_result['realistic'] else 'Unrealistic'

                    if attempt == 1:
                        print(f"   {status} {quality} quality ({compiles}, {realistic})")
                    else:
                        print(f"   {status} {quality} quality ({compiles}, {realistic}) - regenerated")

                    if not validation_result['compiles'] and validation_result['issues']:
                        print(f"   Issues: {len(validation_result['issues'])}")
                        if args.verbose:
                            for issue in validation_result['issues'][:3]:  # Show first 3 issues
                                print(f"     - {issue}")

                    # Store final results
                    final_result = result
                    final_validation = validation_result

                    # Check if we should regenerate
                    if needs_regeneration:
                        print(f"   🔄 Low quality detected, regenerating (attempt {attempt + 1}/{max_attempts})...")
                        regeneration_stats['total_regenerations'] += 1
                        # Remove the low-quality test file so it can be regenerated
                        if os.path.exists(result['test_file']):
                            os.remove(result['test_file'])
                        continue
                    else:
                        # Quality is acceptable or we've reached max attempts
                        break

                except Exception as e:
                    print(f"   ❌ Error processing {rel_path}: {str(e)}")
                    break

            # Process successful results - keep the last successful one
            if successful_results:
                final_result, final_validation = successful_results[-1]  # Keep the last (potentially best) result
                successful_generations += 1
                validation_reports.append(final_validation)

                # Track successful regenerations
                if attempt > 1:
                    regeneration_stats['successful_regenerations'] += 1

                print(f"   ✅ [OK] Final: {os.path.basename(final_result['test_file'])} ({final_validation['quality']} quality)")
            else:
                print(f"   ❌ [ERROR] Failed to generate acceptable test for {rel_path}")

        # Save validation reports
        if validation_reports:
            print(f"\n📊 [POST] Saving validation reports...")
            report_dir = os.path.join(args.repo_path, args.output, "compilation_report")

            for report in validation_reports:
                validator.save_validation_report(report, report_dir)

        # Print summary
        print(f"\n🎉 [DONE] COMPLETED!")
        print(f"   ✅ [OK] Generated: {successful_generations}/{len(c_files)} files")
        print(f"   💾 [SAVE] Tests saved to: {output_dir}")
        if validation_reports:
            print(f"   💾 [SAVE] Reports saved to: {os.path.join(args.output, 'compilation_report')}")

        # Print regeneration statistics
        if args.regenerate_on_low_quality:
            print(f"   🔄 [GEN] Regenerations: {regeneration_stats['successful_regenerations']}/{regeneration_stats['total_regenerations']} successful")
            if regeneration_stats['total_regenerations'] > 0:
                success_rate = (regeneration_stats['successful_regenerations'] / regeneration_stats['total_regenerations']) * 100
                print(f"   🔄 [GEN] Regeneration success rate: {success_rate:.1f}%")

        # Check quality of all generated tests
        quality_levels = {'low': 0, 'medium': 1, 'high': 2}
        threshold_quality_level = quality_levels.get(args.quality_threshold.lower(), 2)

        low_quality_tests = []
        for report in validation_reports:
            current_quality_level = quality_levels.get(report['quality'].lower(), 0)
            if current_quality_level < threshold_quality_level:
                low_quality_tests.append(report['file'])

        if low_quality_tests:
            if args.regenerate_on_low_quality:
                # When regeneration is enabled, warn but don't fail
                print(f"⚠️ [WARN] {len(low_quality_tests)} test(s) still below {args.quality_threshold} quality threshold after regeneration:")
                for test_file in low_quality_tests:
                    print(f"   ⚠️ [WARN] - {test_file}")
                print("💡 [INFO] Consider increasing --max-regeneration-attempts or relaxing --quality-threshold")
            else:
                # When regeneration is disabled, strict enforcement
                print(f"❌ [ERROR] {len(low_quality_tests)} test(s) failed to meet {args.quality_threshold} quality threshold:")
                for test_file in low_quality_tests:
                    print(f"   ❌ [ERROR] - {test_file}")
                print("💡 [INFO] Use --regenerate-on-low-quality to automatically improve test quality")
                sys.exit(1)

        # Overall success check - only fail if no tests were generated at all
        if successful_generations == 0:
            print("❌ [ERROR] No tests were successfully generated")
            sys.exit(1)
        elif successful_generations < len(c_files):
            print(f"⚠️ [WARN] {successful_generations}/{len(c_files)} files successfully generated tests - check validation reports")
            print("💡 [INFO] Some files failed to generate tests (likely due to API timeouts or other issues)")
            # Don't exit with error - allow CI/CD to continue with partial success

        # Print final success message
        print("🎉 [SUCCESS] Test generation completed!")
        print(f"   📊 [STATS] Generated: {successful_generations}/{len(c_files)} files")
        print(f"   📁 [OUTPUT] Tests saved to: {args.output}")
        if validation_reports:
            print(f"   📋 [REPORTS] Validation reports saved to: {args.output}/compilation_report")
        if regeneration_stats['total_regenerations'] > 0:
            success_rate = (regeneration_stats['successful_regenerations'] / regeneration_stats['total_regenerations']) * 100
            print(f"   🔄 [REGEN] Success rate: {success_rate:.1f}% ({regeneration_stats['successful_regenerations']}/{regeneration_stats['total_regenerations']})")

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()