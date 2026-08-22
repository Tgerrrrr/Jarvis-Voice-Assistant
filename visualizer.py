import tkinter as tk
import queue
import threading


class Visualizer:
    """
    Small always-on-top circular pulse pop-up.

    Runs in its own thread (Tkinter needs its own mainloop). Other threads
    talk to it through push_level() and set_state(), both thread-safe.

    States:
      "idle"      - hidden (after a short delay)
      "listening" - shown, blue pulse, driven by mic input levels
      "speaking"  - shown, green pulse, driven by TTS playback levels
    """

    SIZE = 210
    MARGIN_X = 34
    MARGIN_Y = 34

    COLORS = {
        "listening": "#3B8BD4",
        "speaking": "#1D9E75",
        "idle": "#888780",
    }

    BG = "#111111"

    HIDE_DELAY_MS = 1200

    RING_COUNT = 3
    CYCLE_FRAMES = 50  # ~1.7s per ring cycle at 30fps
    MIN_RADIUS = 16
    MAX_RADIUS = 48

    DOT_MIN_RADIUS = 10
    DOT_MAX_RADIUS = 22

    def __init__(self, corner="top-left"):

        self.corner = corner

        self._level_queue = queue.Queue()
        self._state = "idle"
        self._state_lock = threading.Lock()
        self._current_level = 0.0
        self._visible = False
        self._hide_after_id = None
        self._frame = 0

        self._root = None
        self._canvas = None
        self._rings = []
        self._glow = None
        self._dot = None

        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def push_level(self, level):
        """level: float, roughly 0.0-1.0 (RMS amplitude of an audio chunk)."""
        try:
            self._level_queue.put_nowait(min(max(level, 0.0), 1.0))
        except queue.Full:
            pass

    def set_state(self, state):
        """state: 'idle' | 'listening' | 'speaking'"""
        with self._state_lock:
            self._state = state

    # ============================================================
    # INTERNAL - everything below runs on the Tkinter thread
    # ============================================================

    def _run(self):

        self._root = tk.Tk()
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", 0.92)
        self._root.configure(bg="black")

        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()

        if self.corner == "top-left":
            x, y = self.MARGIN_X, self.MARGIN_Y
        elif self.corner == "top-right":
            x, y = screen_w - self.SIZE - self.MARGIN_X, self.MARGIN_Y
        elif self.corner == "bottom-left":
            x, y = self.MARGIN_X, screen_h - self.SIZE - self.MARGIN_Y - 40
        else:  # bottom-right
            x, y = (
                screen_w - self.SIZE - self.MARGIN_X,
                screen_h - self.SIZE - self.MARGIN_Y - 40
            )

        self._root.geometry(f"{self.SIZE}x{self.SIZE}+{x}+{y}")

        self._canvas = tk.Canvas(
            self._root,
            width=self.SIZE,
            height=self.SIZE,
            bg=self.BG,
            highlightthickness=0
        )
        self._canvas.pack()

        cx = self.SIZE / 2
        cy = self.SIZE / 2

        # Rings created first so they render behind the glow and dot
        for _ in range(self.RING_COUNT):

            ring = self._canvas.create_oval(
                cx, cy, cx, cy,
                outline=self.COLORS["idle"],
                width=2
            )

            self._rings.append(ring)

        self._glow = self._canvas.create_oval(
            cx, cy, cx, cy,
            fill=self.COLORS["idle"],
            outline=""
        )

        self._dot = self._canvas.create_oval(
            cx, cy, cx, cy,
            fill=self.COLORS["idle"],
            outline=""
        )

        self._root.withdraw()

        self._tick()
        self._root.mainloop()

    @staticmethod
    def _lerp_color(c1, c2, t):
        """Blend two hex colors. t=0 -> c1, t=1 -> c2."""

        c1 = c1.lstrip("#")
        c2 = c2.lstrip("#")

        r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
        r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)

        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)

        return f"#{r:02x}{g:02x}{b:02x}"

    def _tick(self):

        latest_level = None

        try:
            while True:
                latest_level = self._level_queue.get_nowait()
        except queue.Empty:
            pass

        if latest_level is not None:
            self._current_level = latest_level

        with self._state_lock:
            state = self._state

        if state != "idle":
            self._show()
            self._current_level *= 0.85
        else:
            self._current_level *= 0.8
            self._schedule_hide()

        color = self.COLORS.get(state, self.COLORS["idle"])
        level = self._current_level

        cx = self.SIZE / 2
        cy = self.SIZE / 2

        self._frame += 1

        # Expanding, fading rings - each offset a third of a cycle apart
        for i, ring in enumerate(self._rings):

            offset = i / self.RING_COUNT
            progress = ((self._frame / self.CYCLE_FRAMES) + offset) % 1.0

            size_factor = 0.5 + 0.5 * level
            radius = self.MIN_RADIUS + progress * (self.MAX_RADIUS - self.MIN_RADIUS) * size_factor

            ring_color = self._lerp_color(color, self.BG, progress)

            self._canvas.coords(
                ring,
                cx - radius, cy - radius,
                cx + radius, cy + radius
            )
            self._canvas.itemconfig(ring, outline=ring_color)

        # Core dot pulses in size with the current audio level
        dot_radius = self.DOT_MIN_RADIUS + level * (self.DOT_MAX_RADIUS - self.DOT_MIN_RADIUS)

        glow_radius = dot_radius + 8
        glow_color = self._lerp_color(color, self.BG, 0.55)

        self._canvas.coords(
            self._glow,
            cx - glow_radius, cy - glow_radius,
            cx + glow_radius, cy + glow_radius
        )
        self._canvas.itemconfig(self._glow, fill=glow_color)

        self._canvas.coords(
            self._dot,
            cx - dot_radius, cy - dot_radius,
            cx + dot_radius, cy + dot_radius
        )
        self._canvas.itemconfig(self._dot, fill=color)

        self._root.after(33, self._tick)

    def _show(self):

        if not self._visible:
            self._root.deiconify()
            self._visible = True

        if self._hide_after_id:
            self._root.after_cancel(self._hide_after_id)
            self._hide_after_id = None

    def _schedule_hide(self):

        if self._visible and self._hide_after_id is None:
            self._hide_after_id = self._root.after(self.HIDE_DELAY_MS, self._hide)

    def _hide(self):

        self._root.withdraw()
        self._visible = False
        self._hide_after_id = None