"""Small terminal status indicator; no animation in redirected output."""
import itertools
import sys
import threading
import time


class Status:
    def __init__(self, label, enabled=True):
        self.label = label
        self.enabled = enabled
        self.stream = sys.stdout
        self.animated = enabled and self.stream.isatty()
        self.done = threading.Event()
        self.thread = None
        self.started = None

    def __enter__(self):
        self.started = time.perf_counter()
        if self.animated:
            self.draw("⠋")
            self.thread = threading.Thread(target=self.animate, daemon=True)
            self.thread.start()
        elif self.enabled:
            print(f"{self.label}…", file=self.stream, flush=True)
        return self

    def draw(self, frame):
        elapsed = time.perf_counter() - self.started
        self.stream.write(f"\r\033[2K{frame} {self.label} · {elapsed:.1f}s")
        self.stream.flush()

    def animate(self):
        frames = itertools.cycle("⠙⠹⠸⠼⠴⠦⠧⠇⠏⠋")
        while not self.done.wait(0.1):
            self.draw(next(frames))

    def stop(self):
        if self.done.is_set():
            return
        self.done.set()
        if self.thread:
            self.thread.join()
        if self.animated:
            self.stream.write("\r\033[2K")
            self.stream.flush()

    def __exit__(self, *_):
        self.stop()
