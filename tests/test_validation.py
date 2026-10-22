from src.validation import validate_media_name


def test_image_extensions_are_case_insensitive():
    assert validate_media_name("photo.PNG", "image")[0]


def test_unsupported_media_and_kind_are_rejected():
    assert not validate_media_name("movie.gif", "video")[0]
    assert not validate_media_name("x.jpg", "audio")[0]


def test_missing_extension_is_rejected():
    valid, message = validate_media_name("photo", "image")
    assert not valid
    assert "no extension" in message
