"""
Dependency Analyzer - Analyzes C file dependencies and function relationships
"""

import os
import re
from typing import List, Dict, Set


class DependencyAnalyzer:
    """Analyzes C file dependencies and function relationships"""

    def __init__(self, repo_path: str):
        self.repo_path = os.path.abspath(repo_path)

    def analyze_file_dependencies(self, target_file: str) -> Dict:
        """Comprehensive dependency analysis for any C file"""
        return {
            'file_path': target_file,
            'functions': self._extract_functions(target_file),
            'includes': self._extract_includes(target_file),
            'called_functions': self._find_called_functions(target_file),
            'file_dependencies': self._find_file_dependencies(target_file)
        }

    def _extract_functions(self, file_path: str) -> List[Dict]:
        """Extract function signatures using regex parsing"""
        functions = []
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            # Remove comments and strings for cleaner parsing
            content_clean = re.sub(r'//.*?$|/\*.*?\*/|"(?:\\.|[^"\\])*"', '', content, flags=re.MULTILINE|re.DOTALL)

            # Match function definitions
            pattern = r'(\w+\s*\*?)\s+(\w+)\s*\([^)]*\)\s*\{'
... (truncated)