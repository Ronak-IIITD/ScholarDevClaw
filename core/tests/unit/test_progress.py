import io
import time

import pytest

from scholardevclaw.utils.progress import (
    ProgressConfig,
    ProgressBar,
    progress_iter,
    Spinner,
    MultiProgress,
)


class TestProgressConfig:
    def test_default_config(self):
        config = ProgressConfig()
        assert config.bar_width == 40
        assert config.fill_char == "█"
        assert config.empty_char == "░"
        assert config.show_percent is True
        assert config.show_eta is True

    def test_custom_config(self):
        config = ProgressConfig(
            bar_width=20,
            fill_char="#",
            empty_char="-",
            show_percent=False,
        )
        assert config.bar_width == 20
        assert config.fill_char == "#"
        assert config.empty_char == "-"
        assert config.show_percent is False


class TestProgressBar:
    def test_basic_progress(self):
        stream = io.StringIO()
        config = ProgressConfig(stream=stream, show_eta=False, show_rate=False)
        bar = ProgressBar(100, "Processing", config)

        with bar:
            bar.update(10)

        output = stream.getvalue()
        assert "Processing" in output
        assert "10" in output

    def test_progress_set(self):
        stream = io.StringIO()
        config = ProgressConfig(stream=stream, show_eta=False, show_rate=False)
        bar = ProgressBar(100, config=config)

        with bar:
            bar.set(50)
            bar._last_update = 0
            bar._update_display()

        output = stream.getvalue()
        assert "50" in output

    def test_percent_calculation(self):
        stream = io.StringIO()
        config = ProgressConfig(stream=stream, show_eta=False, show_rate=False, show_count=False)
        bar = ProgressBar(200, config=config)

        with bar:
            bar.set(100)
            bar._last_update = 0
            bar._update_display()

        output = stream.getvalue()
        assert "50.0%" in output

    def test_fill_character(self):
        stream = io.StringIO()
        config = ProgressConfig(
            stream=stream,
            fill_char="=",
            empty_char=".",
            bar_width=10,
            show_eta=False,
            show_rate=False,
            show_count=False,
            show_percent=False,
        )
        bar = ProgressBar(10, config=config)

        with bar:
            bar.set(5)
            bar._last_update = 0
            bar._update_display()

        output = stream.getvalue()
        assert "=====" in output
        assert "....." in output

    def test_context_manager(self):
        stream = io.StringIO()
        config = ProgressConfig(stream=stream, show_eta=False, show_rate=False)
        bar = ProgressBar(100, config=config)

        with bar:
            bar.update(50)

        output = stream.getvalue()
        assert "\n" in output

    def test_progress_iter(self):
        items = [1, 2, 3, 4, 5]
        result = []

        for item in progress_iter(items, "Processing"):
            result.append(item)

        assert result == items

    def test_progress_with_total(self):
        items = [1, 2, 3]
        result = []

        for item in progress_iter(items, "Processing", total=3):
            result.append(item)

        assert result == items


class TestSpinner:
    def test_spinner_basic(self):
        stream = io.StringIO()
        spinner = Spinner("Loading...", stream=stream)

        with spinner:
            spinner.tick()

        output = stream.getvalue()
        assert "Loading..." in output

    def test_spinner_update_message(self):
        stream = io.StringIO()
        spinner = Spinner("Starting...", stream=stream)

        with spinner:
            spinner.tick()
            spinner.update("Processing...")
            spinner.tick()

        output = stream.getvalue()
        assert "Processing..." in output

    def test_spinner_stop_with_message(self):
        stream = io.StringIO()
        spinner = Spinner("Working...", stream=stream)

        with spinner:
            spinner.tick()

        spinner.stop("Done!")
        output = stream.getvalue()
        assert "Done!" in output

    def test_spinner_characters(self):
        stream = io.StringIO()
        spinner = Spinner("Test", stream=stream)

        spinner.start()
        for _ in range(15):
            spinner.tick()
        spinner.stop()

        output = stream.getvalue()
        found = any(char in output for char in Spinner.CHARS)
        assert found


class TestMultiProgress:
    def test_multi_progress_add_bars(self):
        multi = MultiProgress()
        bar1 = multi.add_bar(100, "Task 1")
        bar2 = multi.add_bar(50, "Task 2")

        assert len(multi._bars) == 2
        assert bar1.total == 100
        assert bar2.total == 50

    def test_multi_progress_update(self):
        stream = io.StringIO()
        config = ProgressConfig(stream=stream, show_eta=False, show_rate=False)
        multi = MultiProgress(config=config)

        bar1 = multi.add_bar(10, "Task 1")
        bar2 = multi.add_bar(5, "Task 2")

        bar1.start()
        bar1.update(5)
        bar2.start()
        bar2.update(3)

        multi.close()


class TestTimeFormatting:
    def test_format_seconds(self):
        stream = io.StringIO()
        config = ProgressConfig(stream=stream, show_eta=True, show_rate=False)
        bar = ProgressBar(100, config=config)

        bar._start_time = time.time()
        bar.current = 50
        bar._update_display()

        output = stream.getvalue()
        assert "ETA:" in output or "100" in output

    def test_format_minutes(self):
        stream = io.StringIO()
        config = ProgressConfig(stream=stream)
        bar = ProgressBar(10000, config=config)

        formatted = bar._format_time(125)
        assert "2m" in formatted
        assert "5s" in formatted

    def test_format_hours(self):
        stream = io.StringIO()
        config = ProgressConfig(stream=stream)
        bar = ProgressBar(100000, config=config)

        formatted = bar._format_time(3665)
        assert "1h" in formatted
        assert "1m" in formatted
