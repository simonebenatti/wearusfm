"""Extra probes: the reviewer's broader MU36 variant (`if False:` drops also the full-SHA check)."""
import pytest
import contrastive_release as C
from test_sk1_probes import pkg, Pkg  # noqa: F401


@pytest.mark.parametrize("commit", ["HEAD", "SHORT"])
def test_X_MU36_ref_or_short_commit_refused(pkg, commit):
    p = pkg()
    head = p.review(p.block(p.payload()))
    name = head[:7] if commit == "SHORT" else "HEAD"
    with pytest.raises(ValueError):
        C.bind_review(name, "review.md", p.review_path)


@pytest.mark.parametrize("path", ["../review.md", "ABS", "sub/../review.md"])
def test_X_MU36_uncontained_paths_refused(pkg, path):
    p = pkg()
    head = p.review(p.block(p.payload()))
    target = str(p.review_path) if path == "ABS" else path
    with pytest.raises(ValueError):
        C.bind_review(head, target, p.review_path)
