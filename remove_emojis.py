#!/usr/bin/env python3
"""
Script to remove emojis from all project files
"""
import os
import re
import glob
from pathlib import Path

def remove_emojis_from_file(file_path):
"""Remove emojis from a single file"""
try:
with open(file_path, 'r', encoding='utf-8') as f:
content = f.read()

# Define emoji patterns to remove
emoji_patterns = [
r'', r'', r'', r'', r'', r'', r'', r'', r'', r'',
r'', r'', r'', r'', r'', r'', r'', r'', r'', r'',
r'', r'', r'', r'', r'', r'', r'', r'', r'', r'',
r'', r'', r'', r'', r'', r'', r'', r'', r'', r'',
r'', r'', r'', r''
]

# Remove emojis
modified = False
for pattern in emoji_patterns:
if pattern in content:
content = content.replace(pattern, '')
modified = True

# Clean up extra spaces that might be left
content = re.sub(r' +', ' ', content) # Multiple spaces to single space
content = re.sub(r'^ +', '', content, flags=re.MULTILINE) # Leading spaces on lines

# Write back if modified
if modified:
with open(file_path, 'w', encoding='utf-8') as f:
f.write(content)
print(f"Cleaned: {file_path}")
return True
else:
return False

except Exception as e:
print(f"Error processing {file_path}: {e}")
return False

def main():
"""Remove emojis from all project files"""
# Define file patterns to process
file_patterns = [
"*.py",
"*.md", 
"src/**/*.py",
"configs/*.py",
"notebooks/*.ipynb"
]

project_root = Path('.')
processed_files = 0
modified_files = 0

print("Removing emojis from project files...")

for pattern in file_patterns:
for file_path in project_root.glob(pattern):
if file_path.is_file():
processed_files += 1
if remove_emojis_from_file(file_path):
modified_files += 1

print(f"\nCompleted!")
print(f"Processed: {processed_files} files")
print(f"Modified: {modified_files} files")

if __name__ == "__main__":
main()
