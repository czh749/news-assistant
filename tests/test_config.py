from config import Settings, settings


def test_config_package_exports_settings_instance():
    assert isinstance(settings, Settings)
    assert settings.MYSQL_DATABASE
    assert isinstance(settings.MILVUS_PORT, int)
