import re
from pathlib import Path

README = Path(__file__).parents[1] / "README.md"


def test_readme_example_runs():
    text = README.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    assert blocks, "README has no python example"

    namespace = {}
    exec(blocks[0], namespace)

    assert namespace["phi"].shape == namespace["psi"].shape == (200, 200)
