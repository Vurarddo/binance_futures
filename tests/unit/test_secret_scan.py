from scripts.secret_scan import scan


def test_flags_binance_like_secret(tmp_path):
    f = tmp_path / "x.py"
    f.write_text('k = "' + "Ab1" * 21 + 'Z"\n')
    assert scan(f)


def test_flags_assignment_but_not_empty_or_placeholder(tmp_path):
    f = tmp_path / "e"
    f.write_text("FUT_API_SECRET=abcdefgh12345\n")
    assert scan(f)
    g = tmp_path / "g"
    g.write_text("FUT_API_KEY=\nFUT_API_SECRET=\napi_key: SecretStr | None = None\n")
    assert scan(g) == []


def test_ignores_sha256_hex_and_allow_marker(tmp_path):
    f = tmp_path / "t.py"
    f.write_text(
        'h = "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"\n'
        's = "' + "Q" * 64 + '"  # secret-scan: allow\n'
    )
    assert scan(f) == []
