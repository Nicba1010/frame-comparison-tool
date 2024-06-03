import os.path
import random
import tkinter as tk
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from functools import partial
from pathlib import Path
from tkinter import filedialog
from typing import Tuple, Optional, Dict, List, Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageTk
from loguru import logger

from frame_comparison_tool.components import create_spinbox_frame, create_spinbox_ui


# noinspection PyPep8Naming
@contextmanager
def VideoCapture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


class TextAlignment(Enum):
    LEFT = 'left'
    RIGHT = 'right'


def put_multiline_text(
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        align: TextAlignment = TextAlignment.LEFT,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.66,
        color=(255, 255, 255),
        thickness=1
):
    frame = frame.copy()

    y0, dy = position[1], 30
    for i, line in enumerate(text.splitlines()):
        y = y0 + i * dy

        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]

        cv2.putText(
            frame,
            line,
            (
                position[0] if align == TextAlignment.LEFT else position[0] - text_size[0],
                y
            ),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2
        )
        cv2.putText(
            frame,
            line,
            (
                position[0] if align == TextAlignment.LEFT else position[0] - text_size[0],
                y
            ),
            font,
            font_scale,
            color,
            thickness
        )

    return frame


class FrameType(Enum):
    I_FRAME = 73
    B_FRAME = 66
    P_FRAME = 80
    UNKNOWN_FRAME = 63


def get_frame_type(video_capture: VideoCapture) -> FrameType:
    return FrameType(int(video_capture.get(cv2.CAP_PROP_FRAME_TYPE)))


class ExtractedFrame(object):
    def __init__(
            self,
            frame: np.ndarray,
            frame_type: FrameType,
            frame_number: int,
            source_video_path: Path,
            offset: int,
            additional_text: Optional[Dict[TextAlignment, str]] = None
    ):
        self.frame: np.ndarray = frame
        self.frame_type: FrameType = frame_type
        self.frame_number: int = frame_number
        self.source_video_path: Path = source_video_path
        self.total_number_of_frames: int = get_total_number_of_frames(self.source_video_path)
        self.offset: int = offset
        self.additional_text: Dict[TextAlignment, str] = additional_text or defaultdict(str)

    @property
    def with_overlay(self):
        frame_with_left_overlay = put_multiline_text(
            frame=self.frame,
            text=f"SOURCE: {self.source_video_path.name}\n"
                 f"{self.additional_text[TextAlignment.LEFT]}".strip(),
            position=(10, 30),
            align=TextAlignment.LEFT
        )
        return put_multiline_text(
            frame=frame_with_left_overlay,
            text=f"Frame Type: {self.frame_type.name}\n"
                 f"No.: {self.frame_number}/{self.total_number_of_frames}\n"
                 f"Offset: {self.offset}\n{self.additional_text[TextAlignment.RIGHT]}".strip(),
            position=(frame_with_left_overlay.shape[1] - 10, 30),
            align=TextAlignment.RIGHT
        )

    def __str__(self):
        return f"ExtractedFrame(frame_number={self.frame_number}, frame_type={self.frame_type})"


def get_seconds_per_frame(video_path: Path) -> float:
    with VideoCapture(str(video_path.absolute())) as video_capture:
        return 1 / video_capture.get(cv2.CAP_PROP_FPS)


def get_total_number_of_frames(video_path: Path) -> int:
    with VideoCapture(str(video_path.absolute())) as video_capture:
        return int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))


def extract_closest_wanted_frame(video_path: Path, frame_number: int, offset: int, wanted_frame_type: FrameType):
    logger.info(f"Extracting frame at no. {frame_number} from {video_path} with offset {offset} and wanted frame type "
                f"{wanted_frame_type}")
    with VideoCapture(str(video_path.absolute())) as video_capture:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        original_offset = offset
        while True:
            frame_was_grabbed, current_frame = video_capture.read()

            if not frame_was_grabbed:
                raise RuntimeError(f"Could not extract frame at no. {frame_number} from {video_path}")

            current_frame_number = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            logger.debug(f"Frame number: {current_frame_number}")
            current_frame_type: FrameType = get_frame_type(video_capture)
            logger.debug(f"Frame type: {current_frame_type}")

            if current_frame_type == wanted_frame_type:
                if offset == 0:
                    logger.debug(f'Frame type is correct, offset is 0, returning frame.')
                    return ExtractedFrame(
                        frame=current_frame,
                        frame_type=current_frame_type,
                        frame_number=current_frame_number,
                        source_video_path=video_path,
                        offset=original_offset
                    )
                elif offset > 0:
                    logger.debug(f'Frame type is correct, offset is positive, decrementing offset, searching for next '
                                 f'viable frame.')
                    offset -= 1
                elif offset < 0:
                    logger.debug(f'Frame type is correct, offset is negative, adding a negative offset to timestamp '
                                 f'and searching for next viable frame.')
                    frame_number_offset = 1
                    while True:
                        extracted_frame = extract_closest_wanted_frame(
                            video_path=video_path,
                            frame_number=frame_number - frame_number_offset,
                            offset=0,
                            wanted_frame_type=wanted_frame_type
                        )

                        if extracted_frame.frame_number == current_frame_number:
                            logger.debug(f'Even with the negative timestamp offset, the frame number is the same. '
                                         f'Increasing the negative timestamp offset, trying again.')
                            frame_number_offset += 1
                        else:
                            offset += 1
                            if offset == 0:
                                logger.debug(f'Offset is 0, returning frame.')
                                extracted_frame.offset = original_offset
                                return extracted_frame
                            else:
                                logger.debug(
                                    f'Offset is not 0, decrementing offset, searching for previous viable frame.')
                                current_frame_number = extracted_frame.frame_number
                else:
                    raise ValueError(f"Invalid offset value: {offset}")


def get_next_key(d: Dict[Any, Any], current_key: Any):
    keys: List[Any] = list(d.keys())
    return keys[(keys.index(current_key) + 1) % len(keys)]


def get_prev_key(d: Dict[Any, Any], current_key: Any):
    keys: List[Any] = list(d.keys())
    return keys[(keys.index(current_key) - 1) % len(keys)]


class Comparison(object):
    DEFAULT_SAMPLE_COUNT = 10
    DEFAULT_SEED = 43

    def __init__(
            self,
            wanted_frame_type: FrameType,
            seed: int = DEFAULT_SEED,
            sample_count: int = DEFAULT_SAMPLE_COUNT
    ):
        self.sources: Dict[int, Path] = {}
        self.frames: Dict[int, List[ExtractedFrame]] = {}
        self._shortest_source_video_length: int = self.get_shortest_source_video_length()
        self._wanted_frame_type: FrameType = wanted_frame_type
        self._seed: int = seed
        self._sample_count: int = sample_count
        self._offsets: Dict[int, List[int]] = defaultdict(lambda: [0] * self._sample_count)

        self.current_displayed_source_index: Optional[int] = None
        self.current_displayed_sample_frame_number: Optional[int] = None

        self.on_displayed_sample_frame_change: Optional[Callable[[np.ndarray], None]] = None

    def next_source(self):
        self.current_displayed_source_index = get_next_key(self.sources, self.current_displayed_source_index)
        if self.on_displayed_sample_frame_change:
            self.on_displayed_sample_frame_change(self.get_current_displayed_sample_frame())

    def prev_source(self):
        self.current_displayed_source_index = get_prev_key(self.sources, self.current_displayed_source_index)
        if self.on_displayed_sample_frame_change:
            self.on_displayed_sample_frame_change(self.get_current_displayed_sample_frame())

    def next_frame(self):
        self.current_displayed_sample_frame_number = (
                                                             self.current_displayed_sample_frame_number + 1
                                                     ) % self._sample_count
        if self.on_displayed_sample_frame_change:
            self.on_displayed_sample_frame_change(self.get_current_displayed_sample_frame())

    def prev_frame(self):
        self.current_displayed_sample_frame_number = (
                                                             self.current_displayed_sample_frame_number - 1
                                                     ) % self._sample_count
        if self.on_displayed_sample_frame_change:
            self.on_displayed_sample_frame_change(self.get_current_displayed_sample_frame())

    def get_current_displayed_sample_frame(self):
        return self.frames[self.current_displayed_source_index][self.current_displayed_sample_frame_number].with_overlay

    @property
    def _sample_offsets(self) -> List[int]:
        random.seed(self._seed)  # Use a fixed seed for reproducibility
        return sorted(
            int(random.uniform(0, self._shortest_source_video_length))
            for _
            in range(self._sample_count)
        )

    def set_sample_count(self, sample_count: int):
        self._sample_count = sample_count
        for source_index, offsets in self._offsets.items():
            self._offsets[source_index] = offsets + [0] * (sample_count - len(offsets))
        self.frames.clear()
        self.generate_source_frames()

        if len(self.frames):
            if self.current_displayed_sample_frame_number >= sample_count:
                self.current_displayed_sample_frame_number = sample_count - 1
            if self.on_displayed_sample_frame_change:
                self.on_displayed_sample_frame_change(self.get_current_displayed_sample_frame())

    def set_seed(self, seed: int):
        self._seed = seed
        self._offsets.clear()
        self.frames.clear()
        self.generate_source_frames()
        if len(self.frames) and self.on_displayed_sample_frame_change:
            self.on_displayed_sample_frame_change(self.get_current_displayed_sample_frame())

    def get_shortest_source_video_length(self):
        if not self.sources:
            return int(1e16)
        return min([get_total_number_of_frames(source) for source in self.sources.values()])

    def generate_source_frames(self):
        if self.get_shortest_source_video_length() != self._shortest_source_video_length:
            self._shortest_source_video_length = self.get_shortest_source_video_length()
            self.frames = {}

        for source_index, source_video_path in self.sources.items():
            if source_index not in self.frames:
                self.frames[source_index] = [
                    extract_closest_wanted_frame(
                        video_path=source_video_path,
                        frame_number=sample_offset,
                        offset=self._offsets[source_index][sample_frame_number],
                        wanted_frame_type=self._wanted_frame_type
                    )
                    for sample_frame_number, sample_offset
                    in enumerate(self._sample_offsets)
                ]
            else:
                for sample_frame_number, extracted_frame in enumerate(self.frames[source_index]):
                    if extracted_frame.offset != self._offsets[source_index][sample_frame_number]:
                        self.frames[source_index][sample_frame_number] = extract_closest_wanted_frame(
                            video_path=source_video_path,
                            frame_number=self._sample_offsets[sample_frame_number],
                            offset=self._offsets[source_index][sample_frame_number],
                            wanted_frame_type=self._wanted_frame_type
                        )

    def set_source(self, source_index: int, source_video_path: Path):
        if source_index in self.sources:
            if self.sources[source_index] == source_video_path:
                return
            else:
                self.sources[source_index] = source_video_path
                del self.frames[source_index]
        else:
            self.sources[source_index] = source_video_path

        self.generate_source_frames()

        self.current_displayed_source_index = source_index
        if self.current_displayed_sample_frame_number is None:
            self.current_displayed_sample_frame_number = 0
        if self.on_displayed_sample_frame_change:
            self.on_displayed_sample_frame_change(self.get_current_displayed_sample_frame())

    def set_offset(self, source_index: int, sample_frame_number: int, offset: int):
        self._offsets[source_index][sample_frame_number] = offset
        self.generate_source_frames()
        if self.on_displayed_sample_frame_change:
            self.on_displayed_sample_frame_change(self.get_current_displayed_sample_frame())

    def get_offset(self, source_index: int, sample_frame_number: int) -> int:
        return self.frames[source_index][sample_frame_number].offset

    def save_frames(self, output_directory: Path = Path('../')):
        for source_index, source_video_path in self.sources.items():
            for sample_frame_number, (extracted_frame) in enumerate(self.frames[source_index]):
                output_file_path = output_directory / f"{source_video_path.name}.sample_{sample_frame_number:04d}.png"
                cv2.imwrite(str(output_file_path), extracted_frame.with_overlay)
                logger.info(
                    f"Saved sample frame {sample_frame_number} from source {source_index} to {output_file_path}")


ALLOWED_EXTENSIONS = ("*.mp4 *.avi *.mov *.mkv *.flv *.webm *.wmv *.mpeg *.mpg *.m4v *.3gp *.3g2 *.asf *.vob *.ts "
                      "*.m2ts *.mts *.m2t *.mxf *.ogv *.ogg *.rm *.rmvb *.drc *.yuv *.xvid *.svi *.3gp2 *.3g2 "
                      "*.m2v *.m4v *.m2ts *.mts *.m2t")


def load_video_path(comparison: Comparison, source_index: int, entry_widget: tk.Entry):
    path = filedialog.askopenfilename(filetypes=[("Video files", ALLOWED_EXTENSIONS)])
    if path:
        entry_widget.config(state=tk.NORMAL)
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, path)
        entry_widget.config(state=tk.DISABLED)
        comparison.set_source(source_index, Path(path))


def main():
    comparison: Comparison = Comparison(wanted_frame_type=FrameType.B_FRAME)

    app = tk.Tk()
    app.title("Nicba's Frame Comparison Tool")
    app.minsize(800, 0)  # Set the minimum width to 800px and no minimum height

    app.grid_rowconfigure(0, weight=1)  # Give the image label row a weight
    app.grid_rowconfigure(1, weight=0)  # Give other elements less or no weight if they shouldn't resize much
    app.grid_columnconfigure(0, weight=1)

    image_label = tk.Label(app)  # Label to display the frames
    image_label.grid(row=0, column=0, sticky=tk.NSEW)  # Make the label fill the window

    def display_frame(frame):
        nonlocal image_label
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        image_label.config(image=imgtk)
        image_label.image = imgtk  # Keep a reference, prevent garbage-collection

    comparison.on_displayed_sample_frame_change = lambda frame: display_frame(frame)

    num_sources = int(os.getenv('NUM_SOURCES', 2))  # Get the number of sources from an environment variable

    parameter_frame: tk.Frame = tk.Frame(app)
    parameter_frame.grid(row=1, column=0, sticky='ew')
    # Make sources frame expand and fill all available space
    parameter_frame.grid_columnconfigure(0, weight=1)

    sources_frame = tk.Frame(parameter_frame)
    sources_frame.grid(row=0, column=0, sticky='ew')

    for source_index in range(num_sources):
        frame: tk.Frame = tk.Frame(sources_frame)
        frame.pack(fill=tk.X, expand=True, pady=2)

        tk.Label(frame, text=f"Source {source_index + 1}:").pack(side=tk.LEFT, fill=tk.Y)

        entry_widget = tk.Entry(frame)
        entry_widget.config(state=tk.DISABLED)
        entry_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Button(
            frame,
            text="Browse",
            command=partial(load_video_path, comparison, source_index, entry_widget)
        ).pack(side=tk.LEFT, fill=tk.Y)

    # Adding the Comparison Count and Seed field
    comparison_and_seed_frame = tk.Frame(parameter_frame)
    comparison_and_seed_frame.grid(row=1, column=0, sticky='ew')

    sample_count_spinbox_frame: tk.Frame = create_spinbox_frame(
        comparison_and_seed_frame,
        "Sample Count:",
        1,
        50,
        1,
        Comparison.DEFAULT_SAMPLE_COUNT,
        lambda spinbox: comparison.set_sample_count(int(spinbox.get()))
    )
    sample_count_spinbox_frame.pack(side=tk.LEFT)

    seed_spinbox_frame: tk.Frame = create_spinbox_frame(
        comparison_and_seed_frame,
        "Sample Count:",
        1,
        50,
        1,
        Comparison.DEFAULT_SEED,
        lambda spinbox: comparison.set_seed(int(spinbox.get()))
    )
    seed_spinbox_frame.pack(side=tk.RIGHT)

    create_spinbox_ui(
        parent=parameter_frame,
        text="Crop",
        callback=lambda top, bottom, left, right: logger.debug(
            f"Top: {top}, Bottom: {bottom}, Left: {left}, Right: {right}"),
    ).grid(row=0, column=1, rowspan=2, sticky='ew')

    create_spinbox_ui(
        parent=parameter_frame,
        text="Scale",
        callback=lambda top, bottom, left, right: logger.debug(
            f"Top: {top}, Bottom: {bottom}, Left: {left}, Right: {right}"),
    ).grid(row=0, column=2, rowspan=2, sticky='ew')

    app.bind("<Left>", lambda event: comparison.prev_source())
    app.bind("<Right>", lambda event: comparison.next_source())
    app.bind("<Up>", lambda event: comparison.prev_frame())
    app.bind("<Down>", lambda event: comparison.next_frame())
    app.bind(
        "<Insert>",
        lambda event: comparison.set_offset(
            comparison.current_displayed_source_index,
            comparison.current_displayed_sample_frame_number,
            comparison.get_offset(
                comparison.current_displayed_source_index,
                comparison.current_displayed_sample_frame_number
            ) + 1
        )
    )
    app.bind(
        "<Delete>",
        lambda event: comparison.set_offset(
            comparison.current_displayed_source_index,
            comparison.current_displayed_sample_frame_number,
            comparison.get_offset(
                comparison.current_displayed_source_index,
                comparison.current_displayed_sample_frame_number
            ) - 1
        )
    )
    app.bind(
        "<Control-Left>",
        lambda event: comparison.save_frames()
    )

    def maximize_if_too_big(event):
        app.state(
            'zoomed'
            if app.winfo_width() >= app.winfo_screenwidth() or app.winfo_height() >= app.winfo_screenheight()
            else 'normal'
        )

    def keep_inside_screen(event):
        # Get screen width and height
        screen_width = app.winfo_screenwidth()
        screen_height = app.winfo_screenheight()

        # Get current window width, height, and position
        window_width = app.winfo_width()
        window_height = app.winfo_height()
        x = app.winfo_x()
        y = app.winfo_y()

        # Adjust x coordinate if window goes off the screen edges
        if x + window_width > screen_width:
            x = screen_width - window_width
        if x < 0:
            x = 0

        # Adjust y coordinate if window goes off the screen edges
        if y + window_height > screen_height:
            y = screen_height - window_height
        if y < 0:
            y = 0

        # Move window only if adjustments were made
        if x != app.winfo_x() or y != app.winfo_y():
            app.geometry(f'+{x}+{y}')

    def handle_configure(event):
        maximize_if_too_big(event)
        # keep_inside_screen(event)

    app.bind("<Control-s>", lambda event: comparison.save_frames())
    app.bind('<Configure>', lambda event: handle_configure(event))

    app.mainloop()


if __name__ == "__main__":
    main()
