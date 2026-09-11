import re, pathlib

suspicious = re.compile(
    r'(?i)(api_key|password|secret|token)\s*=\s*[A-Za-z0-9_\-]{20,}'
)
checked = []
issues = []
dirs = ['src', 'dashboard']
for d in dirs:
    for f in pathlib.Path(d).rglob('*.py'):
        text = f.read_text(encoding='utf-8', errors='ignore')
        for m in suspicious.finditer(text):
            issues.append(f'{f}: {m.group()[:80]}')
        checked.append(str(f))

print(f'Files scanned: {len(checked)}')
if issues:
    print('POTENTIAL HARDCODED SECRETS:')
    for i in issues:
        print(' ', i)
else:
    print('No hardcoded secrets found. PASSED.')
