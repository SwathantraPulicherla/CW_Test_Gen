"""
Dependency Analyzer - Analyzes C++ file dependencies and function relationships
"""

import os
import re
from typing import List, Dict, Set


class DependencyAnalyzer:
    """Analyzes C++ file dependencies and function relationships"""

    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)

    def analyze_file_dependencies(self, target_file: str) -> Dict:
        """Comprehensive dependency analysis for any C++ file"""
        return {
            'file_path': target_file,
            'functions': self._extract_functions(target_file),
            'includes': self._extract_includes(target_file),
            'called_functions': self._find_called_functions(target_file),
            'file_dependencies': self._find_file_dependencies(target_file)
        }

    def find_all_c_files(self) -> List[str]:
        """Find all C/C++ source files in the repository"""
        c_files = []
        for root, dirs, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith(('.cpp', '.cc', '.cxx', '.c++', '.c', '.ino')):
                    c_files.append(os.path.join(root, file))
        return c_files

    def _extract_functions(self, file_path: str) -> List[Dict]:
        """Extract function signatures using regex parsing"""
        functions = []
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Remove comments and strings for cleaner parsing
            content_clean = re.sub(r'//.*?$|/\*.*?\*/|"(?:\\.|[^"\\])*"', '', content, flags=re.MULTILINE|re.DOTALL)

            # Match function definitions (C and C++)
            pattern = r'(\w+(?:\s*\*|\s*::\s*\w+)?)\s+(\w+(?:::\w+)*)\s*\([^)]*\)\s*\{'
            matches = re.finditer(pattern, content_clean, re.MULTILINE)

            for match in matches:
                return_type = match.group(1).strip()
                func_name = match.group(2).strip()
                # Extract parameters (simplified)
                param_start = content_clean.find('(', match.end())
                param_end = content_clean.find(')', param_start)
                params = content_clean[param_start:param_end+1] if param_start != -1 and param_end != -1 else '()'

                functions.append({
                    'name': func_name,
                    'signature': f'{return_type} {func_name}{params}',
                    'return_type': return_type,
                    'parameters': params
                })

        except Exception as e:
            print(f'Error extracting functions from {file_path}: {e}')

        return functions

    def _extract_includes(self, file_path: str) -> List[str]:
        includes = []
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            includes = re.findall(r'#include\s+[<"]([^>"]+)[>"]', content)
        except Exception as e:
            print(f'Error extracting includes from {file_path}: {e}')
        return includes

    def _find_called_functions(self, file_path: str) -> List[str]:
        called = []
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            # Simple regex for function calls
            called = re.findall(r'\b(\w+)\s*\(', content)
            # Filter out keywords
            keywords = {'if', 'for', 'while', 'switch', 'return', 'sizeof', 'malloc', 'free'}
            called = [c for c in called if c not in keywords]
        except Exception as e:
            print(f'Error finding called functions in {file_path}: {e}')
        return called

    def _find_file_dependencies(self, file_path: str) -> List[str]:
        deps = []
        includes = self._extract_includes(file_path)
        for inc in includes:
            if not inc.startswith('<'):  # Local includes
                dep_path = os.path.join(os.path.dirname(file_path), inc)
                if os.path.exists(dep_path):
                    deps.append(dep_path)
        return deps