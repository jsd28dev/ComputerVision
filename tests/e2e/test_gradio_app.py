"""End-to-end tests for the Gradio app.

Three layers, cheapest first, so a failure localises itself:

1. ``DetectionService`` — the logic, with no Gradio import at all.
2. ``gradio_client`` — the app running as a real HTTP server, driven through
   its ``/detect`` endpoint. Catches wiring bugs (wrong input order, wrong
   output count) without a browser.
3. Playwright — a real browser clicking real controls. Catches only what the
   other two cannot: that the components render and are actually operable.

The browser layer selects on the ``elem_id`` values in
``smalldet.app.gradio_app.ELEM_IDS``. Gradio's own generated ids change between
releases and layout edits, so those constants are the testing contract.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _support import (  # noqa: E402
    expect_error,
    has_module,
    require_module,
    skip,
    synthetic_dataset,
    tiny_config,
)

_SERVER = {"demo": None, "url": None}


# --------------------------------------------------------------- layer 1: logic


def _service():
    """A DetectionService over a randomly-initialised model.

    The predictions are meaningless — what is under test is that the service
    formats, thresholds, and renders them correctly.
    """
    from smalldet.app.service import DetectionService
    from smalldet.inference import Predictor
    from smalldet.models import build_model
    from smalldet.visualization import Renderer

    config = tiny_config()
    names = ["__background__", "screw", "washer", "nut"]
    predictor = Predictor(build_model(config.model, 4), names, config.predict)
    return DetectionService(predictor, Renderer(config.visualize, names), config)


def test_service_returns_image_summary_and_records():
    service = _service()
    path = sorted(synthetic_dataset()["images"].glob("*.png"))[0]

    image, summary, records = service.detect(str(path), score_threshold=0.0)

    assert image is not None and image.size[0] > 0
    assert isinstance(summary, str) and summary
    assert isinstance(records, list)
    for record in records:
        assert set(record) == {"label", "class_name", "score", "box_xyxy", "area"}


def test_service_handles_no_image_without_crashing():
    """Gradio hands over None whenever the user clicks Detect on an empty box."""
    image, summary, records = _service().detect(None)
    assert image is None
    assert records == []
    assert "upload" in summary.lower()


def test_raising_the_threshold_reduces_detections():
    service = _service()
    path = str(sorted(synthetic_dataset()["images"].glob("*.png"))[0])

    _, _, permissive = service.detect(path, score_threshold=0.0)
    _, _, strict = service.detect(path, score_threshold=0.99)
    assert len(strict) <= len(permissive)


def test_summary_reports_the_same_buckets_the_metrics_use():
    """What the UI shows and what AP_small / AP_medium measure must be the same
    partition, or the demo tells a different story than the evaluation."""
    service = _service()
    path = str(sorted(synthetic_dataset()["images"].glob("*.png"))[0])
    _, summary, _ = service.detect(path, score_threshold=0.0)

    if "No detections" not in summary:
        assert "small" in summary and "medium" in summary and "large" in summary
        assert "32²" in summary


def test_model_summary_describes_what_is_loaded():
    summary = _service().model_summary()
    assert "fasterrcnn_resnet50_fpn_v2" in summary
    assert "Anchors" in summary  # the small-object setting is surfaced
    assert "screw" in summary


def test_examples_resolution_skips_missing_paths():
    from smalldet.app.service import resolve_examples

    config = tiny_config(
        app={"examples": [str(synthetic_dataset()["images"]), "does/not/exist.png"]}
    )
    resolved = resolve_examples(config)
    assert resolved
    assert all(Path(path).is_file() for path in resolved)


# ------------------------------------------------------- the server under test


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _launch_server() -> str:
    """Start the app once per process and reuse it across tests."""
    if _SERVER["url"]:
        return _SERVER["url"]

    require_module("gradio", "the app tests need it")

    from smalldet.app.gradio_app import build_interface
    from smalldet.inference import Predictor
    from smalldet.models import build_model
    from smalldet.visualization import Renderer

    config = tiny_config(
        app={
            "examples": [],
            "server_name": "127.0.0.1",
            "title": "smalldet test app",
        },
        predict={"postprocess": {"score_threshold": 0.0, "max_detections": 20}},
    )
    names = ["__background__", "screw", "washer", "nut"]
    predictor = Predictor(build_model(config.model, 4), names, config.predict)
    demo = build_interface(config, predictor, Renderer(config.visualize, names))

    port = _free_port()
    demo.queue(default_concurrency_limit=1)
    thread = threading.Thread(
        target=lambda: demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            prevent_thread_lock=True,
            show_error=True,
            quiet=True,
        ),
        daemon=True,
    )
    thread.start()

    url = f"http://127.0.0.1:{port}"
    for _ in range(120):  # the model build dominates startup
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.5)
    else:
        raise RuntimeError(f"the Gradio app did not come up on {url}")

    _SERVER.update({"demo": demo, "url": url})
    return url


# ------------------------------------------------------- layer 2: the HTTP API


def test_app_exposes_the_detect_endpoint():
    """view_api() is the contract a client depends on; never guess at it."""
    require_module("gradio_client")
    from gradio_client import Client

    client = Client(_launch_server(), verbose=False)
    api = client.view_api(return_format="dict")
    endpoints = api.get("named_endpoints", {})
    assert "/detect" in endpoints, f"available: {list(endpoints)}"

    parameters = endpoints["/detect"]["parameters"]
    # image, score_threshold, max_detections, use_tiling, highlight_small
    assert len(parameters) == 5
    assert len(endpoints["/detect"]["returns"]) == 3


def test_detect_endpoint_returns_image_summary_and_json():
    require_module("gradio_client")
    from gradio_client import Client, handle_file

    client = Client(_launch_server(), verbose=False)
    path = str(sorted(synthetic_dataset()["images"].glob("*.png"))[0])

    image, summary, records = client.predict(
        handle_file(path), 0.0, 20, False, True, api_name="/detect"
    )
    assert image is not None
    assert isinstance(summary, str) and summary
    assert isinstance(records, list)


def test_detect_endpoint_honours_the_threshold():
    require_module("gradio_client")
    from gradio_client import Client, handle_file

    client = Client(_launch_server(), verbose=False)
    path = str(sorted(synthetic_dataset()["images"].glob("*.png"))[0])

    _, _, permissive = client.predict(
        handle_file(path), 0.0, 20, False, True, api_name="/detect"
    )
    _, _, strict = client.predict(
        handle_file(path), 0.999, 20, False, True, api_name="/detect"
    )
    assert len(strict) <= len(permissive)


def test_tiled_inference_runs_through_the_api():
    require_module("gradio_client")
    from gradio_client import Client, handle_file

    client = Client(_launch_server(), verbose=False)
    path = str(sorted(synthetic_dataset()["images"].glob("*.png"))[0])
    image, summary, _ = client.predict(
        handle_file(path), 0.0, 20, True, True, api_name="/detect"
    )
    assert image is not None and summary


# --------------------------------------------------------- layer 3: the browser


def _page(browser, url: str):
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_selector("#sd-detect-button", timeout=60_000)
    return page


def test_browser_renders_every_declared_control():
    """The elem_id values are the contract these tests select on.

    Only the Detect page's ids are checked here: Gradio mounts a tab's contents
    lazily, so the ``ft_*`` controls do not exist until the Finetune tab is
    opened. They are covered by test_browser_finetune_page_renders_every_control.
    """
    require_module("playwright", "run `playwright install chromium` after installing")
    from playwright.sync_api import sync_playwright

    from smalldet.app.gradio_app import ELEM_IDS

    detect_ids = {k: v for k, v in ELEM_IDS.items() if not k.startswith("ft_")}
    url = _launch_server()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = _page(browser, url)
            for name, elem_id in detect_ids.items():
                assert page.locator(f"#{elem_id}").count() == 1, f"missing {name}"
        finally:
            browser.close()


def test_browser_detect_flow_produces_an_overlay_and_a_summary():
    """The full user journey: upload, click Detect, see boxes and a summary."""
    require_module("playwright")
    from playwright.sync_api import sync_playwright

    url = _launch_server()
    path = str(sorted(synthetic_dataset()["images"].glob("*.png"))[0])

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = _page(browser, url)
            page.set_input_files("#sd-input-image input[type=file]", path)
            page.wait_for_selector("#sd-input-image img", timeout=30_000)

            page.click("#sd-detect-button button, #sd-detect-button")
            page.wait_for_selector("#sd-output-image img", timeout=180_000)

            summary = page.inner_text("#sd-summary")
            assert "detection" in summary.lower()
            # The size breakdown must reach the browser, not just the service.
            assert "small" in summary.lower()
        finally:
            browser.close()


def test_browser_threshold_slider_changes_the_result():
    require_module("playwright")
    from playwright.sync_api import sync_playwright

    url = _launch_server()
    path = str(sorted(synthetic_dataset()["images"].glob("*.png"))[0])

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = _page(browser, url)
            page.set_input_files("#sd-input-image input[type=file]", path)
            page.wait_for_selector("#sd-input-image img", timeout=30_000)

            def detect(threshold: str) -> str:
                box = page.locator("#sd-score-threshold input[type=number]")
                box.fill(threshold)
                box.press("Enter")
                page.click("#sd-detect-button button, #sd-detect-button")
                # Wait on the threshold value alone. The summary has two
                # branches — a detections table, or "No detections above a
                # score of X" — and a high threshold legitimately produces the
                # second, so anything phrasing-specific makes this flaky.
                page.wait_for_function(
                    "t => (document.querySelector('#sd-summary')?.innerText || '')"
                    "       .includes(t)",
                    arg=f"{float(threshold):.2f}",
                    timeout=180_000,
                )
                return page.inner_text("#sd-summary")

            permissive = detect("0.00")
            strict = detect("0.99")
            assert permissive != strict
            assert "0.99" in strict
        finally:
            browser.close()


def test_browser_clear_button_resets_the_panel():
    require_module("playwright")
    from playwright.sync_api import sync_playwright

    url = _launch_server()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = _page(browser, url)
            page.click("#sd-clear-button button, #sd-clear-button")
            page.wait_for_function(
                "() => document.querySelector('#sd-summary')"
                "  ?.innerText.toLowerCase().includes('upload an image')",
                timeout=30_000,
            )
        finally:
            browser.close()


def test_finetune_page_exposes_its_endpoints():
    """The finetuning page's buttons must reach real functions, not stubs."""
    require_module("gradio_client")
    from gradio_client import Client

    client = Client(_launch_server(), verbose=False)
    endpoints = client.view_api(return_format="dict").get("named_endpoints", {})
    for name in ("/validate_finetune", "/split_dataset", "/finetune", "/stop_finetune"):
        assert name in endpoints, f"available: {list(endpoints)}"


def test_validate_endpoint_returns_a_plan_or_a_named_error():
    require_module("gradio_client")
    from gradio_client import Client

    client = Client(_launch_server(), verbose=False)
    endpoints = client.view_api(return_format="dict")["named_endpoints"]
    # The control list is long and positional; drive it with the defaults the
    # page was built with, which is exactly what a user pressing Validate does.
    defaults = [p.get("parameter_default") for p in endpoints["/validate_finetune"]["parameters"]]
    text = client.predict(*defaults, api_name="/validate_finetune")
    assert isinstance(text, str)
    assert "Ready to train" in text or "Configuration error" in text


def test_browser_finetune_page_renders_every_control():
    require_module("playwright")
    from playwright.sync_api import sync_playwright

    from smalldet.app.gradio_app import ELEM_IDS

    finetune_ids = {k: v for k, v in ELEM_IDS.items() if k.startswith("ft_")}
    assert len(finetune_ids) > 30, "the finetune page should expose its controls"

    url = _launch_server()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = _page(browser, url)
            page.get_by_role("tab", name="Finetune").click()
            page.wait_for_selector(f"#{ELEM_IDS['ft_strategy']}", timeout=30_000)
            for name, elem_id in finetune_ids.items():
                assert page.locator(f"#{elem_id}").count() >= 1, f"missing {name}"
        finally:
            browser.close()


def test_browser_all_four_strategies_are_offered():
    """The four finetuning modes are the point of the page."""
    require_module("playwright")
    from playwright.sync_api import sync_playwright

    url = _launch_server()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = _page(browser, url)
            page.get_by_role("tab", name="Finetune").click()
            page.wait_for_selector("#sd-ft-strategy", timeout=30_000)
            values = page.eval_on_selector_all(
                "#sd-ft-strategy input", "els => els.map(e => e.value)"
            )
            assert set(values) == {"head_only", "partial", "gradual", "full"}
        finally:
            browser.close()


def test_browser_validate_button_produces_a_plan():
    """Clicking Validate must call the real service and render its verdict."""
    require_module("playwright")
    from playwright.sync_api import sync_playwright

    url = _launch_server()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = _page(browser, url)
            page.get_by_role("tab", name="Finetune").click()
            page.wait_for_selector("#sd-ft-validate-button", timeout=30_000)

            page.click("#sd-ft-validate-button button, #sd-ft-validate-button")
            page.wait_for_function(
                "() => { const t = document.querySelector('#sd-ft-preview')?.innerText || '';"
                "        return t.includes('Ready to train') || t.includes('Configuration error'); }",
                timeout=60_000,
            )
            text = page.inner_text("#sd-ft-preview")
            assert "Ready to train" in text
            # The plan must state the consequences of the strategy, not just echo it.
            assert "Strategy" in text and "Optimizer" in text
        finally:
            browser.close()


def test_browser_validate_surfaces_a_bad_setting_instead_of_crashing():
    require_module("playwright")
    from playwright.sync_api import sync_playwright

    url = _launch_server()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = _page(browser, url)
            page.get_by_role("tab", name="Finetune").click()
            page.wait_for_selector("#sd-ft-anchor-sizes", timeout=30_000)

            box = page.locator("#sd-ft-anchor-sizes textarea, #sd-ft-anchor-sizes input")
            box.first.fill("eight, sixteen")
            page.click("#sd-ft-validate-button button, #sd-ft-validate-button")
            page.wait_for_function(
                "() => (document.querySelector('#sd-ft-preview')?.innerText || '')"
                "        .includes('Configuration error')",
                timeout=60_000,
            )
        finally:
            browser.close()


def test_browser_stop_button_is_wired():
    require_module("playwright")
    from playwright.sync_api import sync_playwright

    url = _launch_server()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = _page(browser, url)
            page.get_by_role("tab", name="Finetune").click()
            page.wait_for_selector("#sd-ft-stop-button", timeout=30_000)

            page.click("#sd-ft-stop-button button, #sd-ft-stop-button")
            page.wait_for_function(
                "() => (document.querySelector('#sd-ft-status')?.innerText || '')"
                "        .toLowerCase().includes('stop')",
                timeout=60_000,
            )
        finally:
            browser.close()


def test_browser_reports_no_console_errors():
    """A page that renders but logs exceptions is broken in a way screenshots
    do not show."""
    require_module("playwright")
    from playwright.sync_api import sync_playwright

    url = _launch_server()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            errors = []
            page = browser.new_page()
            page.on(
                "console",
                lambda message: errors.append(message.text)
                if message.type == "error"
                else None,
            )
            page.goto(url, wait_until="networkidle")
            page.wait_for_selector("#sd-detect-button", timeout=60_000)
            # Favicon 404s and analytics blocks are noise, not app failures.
            real = [
                text
                for text in errors
                if "favicon" not in text.lower() and "analytics" not in text.lower()
            ]
            assert not real, f"console errors: {real}"
        finally:
            browser.close()
