from motac.core.run_dir import make_run_dir

def test_run_dir_deterministic(tmp_path):
    cfg = {'a': 1}
    r1 = make_run_dir(tmp_path, seed=1, cfg=cfg, repo_root='.')
    r2 = make_run_dir(tmp_path, seed=1, cfg=cfg, repo_root='.')
    assert r1 == r2
    assert (r1 / 'metadata.json').exists()
