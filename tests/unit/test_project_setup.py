import app


def test_application_package_is_importable() -> None:
    assert app.__name__ == "app"
