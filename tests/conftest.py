def pytest_addoption(parser):
    parser.addoption(
        "--update-baseline",
        action="store_true",
        help="サンプル写真の現在の結果を回帰チェックの基準として保存する（-m samples と併用）",
    )
