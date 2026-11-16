"""演示界面的单请求锁，以及 app 导入链不带 torch。"""

import threading

import qwen_vlm.app.app as app_module
from qwen_vlm.app.gate import BusyError, InferenceGate


def test_gate_rejects_overlapping_acquire():
    gate = InferenceGate()
    assert gate.try_acquire()
    assert gate.try_acquire() is False
    gate.release()
    assert gate.try_acquire()
    gate.release()


def test_gate_context_manager_is_exclusive():
    gate = InferenceGate()
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def hold():
        try:
            with gate:
                entered.set()
                release.wait(2)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=hold)
    thread.start()
    assert entered.wait(2)
    try:
        with gate:
            raise AssertionError("第二个请求不应该拿到锁")
    except BusyError:
        pass
    release.set()
    thread.join(2)
    assert errors == []
    with gate:
        assert True


def test_app_module_does_not_import_torch():
    source = open(app_module.__file__, encoding="utf-8").read()
    assert "import torch" not in source
    assert "import transformers" not in source
    runner_source = open(app_module.__file__.replace("/app/app.py", "/inference/llamacpp_runner.py"), encoding="utf-8").read()
    assert "import torch" not in runner_source
    assert "import transformers" not in runner_source
    assert "同一时间只处理一个请求" in source
