import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    tests = sorted((root / "tests").glob("validate_*.py"))

    if not tests:
        raise RuntimeError("No validation scripts found.")

    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONPATH"] = str(root)

    for test in tests:
        print(f"\nRunning {test.name}...", flush=True)

        result = subprocess.run(
            [sys.executable, str(test)],
            cwd=root,
            env=env,
        )

        if result.returncode != 0:
            print(f"\nFAILED: {test.name}")
            return result.returncode

        print(f"PASSED: {test.name}")

    print(f"\nALL {len(tests)} VALIDATIONS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
