import subprocess
import sys


def test_skill_validate_runs():
    """Run the simple skill validator and assert it exits 0"""
    ret = subprocess.run([sys.executable, "skills/training-lifecycle/scripts/validate_skill.py", "skills/training-lifecycle"], capture_output=True)
    assert ret.returncode == 0, f"validator failed: {ret.stdout.decode()} {ret.stderr.decode()}"
