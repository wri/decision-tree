import os

def pytest_collection_modifyitems(config, items):
    if os.getenv("SKIP_REMOTE_TESTS") == "true":
        # Mark or deselect tests from remote repository
        deselected = [item for item in items if "external_repo" in str(item.fspath)]
        for item in deselected:
            items.remove(item)
        config.hook.pytest_deselected(items=deselected)