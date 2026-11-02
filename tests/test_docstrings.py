"""Every public function, method and class in changegate states what it does not establish."""

import importlib
import inspect
import pkgutil

import changegate

MARKER = "Does not establish"


def public_objects():
    for info in pkgutil.walk_packages(changegate.__path__, "changegate."):
        module = importlib.import_module(info.name)
        for name, obj in vars(module).items():
            if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
                continue
            if inspect.isfunction(obj):
                yield f"{module.__name__}.{name}", obj
            elif inspect.isclass(obj):
                yield f"{module.__name__}.{name}", obj
                for attr, member in vars(obj).items():
                    if attr.startswith("_") and attr != "__init__":
                        continue
                    target = member.fget if isinstance(member, property) else member
                    target = getattr(target, "__func__", target)
                    # skip generated members (dataclass and Protocol __init__)
                    if (
                        inspect.isfunction(target)
                        and target.__code__.co_filename == module.__file__
                    ):
                        yield f"{module.__name__}.{name}.{attr}", target


def test_every_public_object_states_its_limits():
    missing = [qual for qual, obj in public_objects() if MARKER not in (obj.__doc__ or "")]
    assert missing == []


def test_the_walk_finds_the_interfaces():
    names = {qual for qual, _ in public_objects()}
    for expected in (
        "changegate.checks.Check.run",
        "changegate.battery.Battery.score",
        "changegate.artifacts.ArtifactRef.from_path",
        "changegate.log.ActionLog.append",
        "changegate.record.Roles.single_person",
        "changegate.config.load_config",
    ):
        assert expected in names
