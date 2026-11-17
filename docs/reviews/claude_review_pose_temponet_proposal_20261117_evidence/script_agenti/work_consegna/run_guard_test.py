"""Run the single long-evaluation source-pin test against a tree copy (reviewer demo)."""
import sys
from pathlib import Path
tree = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(tree / 'src'))
import pytest
sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', '--rootdir', str(tree),
    str(tree / 'tests/cpu/test_consolidation_long_common.py') + '::test_concrete_campaign_remains_review_gated_and_all_endpoints_pinned']))
