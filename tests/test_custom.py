import pytest

from chemcrow.tools.custom import CustomTool


@pytest.fixture
def custom_tool():
    return CustomTool()


def test_custom_tool_placeholder(custom_tool):
    # Scaffold assertion -- update once CustomTool._run() has real logic.
    ans = custom_tool._run("CC(=O)Oc1ccccc1C(=O)O")
    assert "scaffold" in ans.lower()
