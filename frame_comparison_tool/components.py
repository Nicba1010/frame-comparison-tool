import tkinter as tk
from functools import partial
from tkinter import Widget, Frame
from typing import Tuple, Callable


def create_spinbox_frame(
        parent: tk.Widget,
        label_text: str,
        from_: int,
        to: int,
        increment: int,
        default: int,
        command: Callable[[tk.Spinbox], None]
) -> tk.Frame:
    frame = tk.Frame(parent)
    frame.pack(pady=2)
    tk.Label(frame, text=label_text).pack(side=tk.LEFT, fill=tk.Y)
    spinbox = tk.Spinbox(
        frame,
        from_=from_,
        to=to,
        increment=increment,
        wrap=True,
        width=5
    )
    spinbox.config(command=partial(command, spinbox))
    spinbox.delete(0, tk.END)
    spinbox.insert(0, str(default))
    spinbox.pack(side=tk.RIGHT)
    return frame


def create_spinbox_ui(
        parent: Widget,
        text: str,
        callback: Callable[[int, int, int, int], None],
        tb_range: Tuple[int, int] = (-400, 400),
        lr_range: Tuple[int, int] = (-400, 400),
) -> Frame:
    """
    Create a spinbox UI with customizable range values for top/bottom and left/right spinboxes.
    Includes a callback function that is invoked whenever any spinbox value changes and passes all values.

    Args:
    parent: The parent widget or window where this UI will be placed.
    text: The text to be displayed on the label.
    tb_range: Tuple (from_, to) representing the range for the top and bottom spinboxes.
    lr_range: Tuple (from_, to) representing the range for the left and right spinboxes.
    callback: Function to be called when any spinbox value changes, passing all spinbox values.
    """
    frame = tk.Frame(parent)

    # Add label on top left
    label = tk.Label(frame, text=f"{text}:")
    label.grid(row=0, column=0, sticky=tk.W)

    # Add little lock icon button on top right 🔒
    lock_button = tk.Button(frame, text="🔒")
    lock_button.grid(row=0, column=2, sticky=tk.EW)

    # Positioning the button at the center
    button = tk.Button(frame, text="RESET")
    button.grid(row=1, column=1, sticky=tk.EW, padx=1, pady=1)

    # Creating spinboxes around the central button with customizable ranges, minimum width
    spinbox_top = tk.Spinbox(frame, from_=tb_range[0], to=tb_range[1], width=6)
    spinbox_top.delete(0, tk.END)
    spinbox_top.insert(0, "0")
    spinbox_top.grid(row=0, column=1)

    spinbox_bottom = tk.Spinbox(frame, from_=tb_range[0], to=tb_range[1], width=6)
    spinbox_bottom.delete(0, tk.END)
    spinbox_bottom.insert(0, "0")
    spinbox_bottom.grid(row=2, column=1)

    spinbox_left = tk.Spinbox(frame, from_=lr_range[0], to=lr_range[1], width=6)
    spinbox_left.delete(0, tk.END)
    spinbox_left.insert(0, "0")
    spinbox_left.grid(row=1, column=0)

    spinbox_right = tk.Spinbox(frame, from_=lr_range[0], to=lr_range[1], width=6)
    spinbox_right.delete(0, tk.END)
    spinbox_right.insert(0, "0")
    spinbox_right.grid(row=1, column=2)

    # State variable for lock state
    lock_state = True

    # Function to toggle lock state and button text
    def toggle_lock():
        nonlocal lock_state
        lock_state = not lock_state
        lock_button.config(text="🔒" if lock_state else "🔓")

    lock_button.config(command=toggle_lock)

    # Function to trigger callback with synchronization when locked
    def trigger_callback():
        nonlocal lock_state
        values = {
            spinbox_top: int(spinbox_top.get()),
            spinbox_bottom: int(spinbox_bottom.get()),
            spinbox_left: int(spinbox_left.get()),
            spinbox_right: int(spinbox_right.get())
        }
        if lock_state:
            # If a spinbox value changes, adjust the opposite spinbox by the same delta
            changed_spinbox = None
            for spinbox in (spinbox_top, spinbox_bottom, spinbox_left, spinbox_right):
                if spinbox.get() != spinbox.last_value:
                    changed_spinbox = spinbox
                    break
            if changed_spinbox:
                delta = values[changed_spinbox] - changed_spinbox.last_value
                opposite_spinbox = (spinbox_bottom if changed_spinbox == spinbox_top else spinbox_top) \
                    if changed_spinbox in (spinbox_top, spinbox_bottom) \
                    else (spinbox_right if changed_spinbox == spinbox_left else spinbox_left)

                new_value = int(opposite_spinbox.get()) + delta
                opposite_spinbox.delete(0, tk.END)
                opposite_spinbox.insert(0, str(new_value))

        callback(
            int(spinbox_top.get()),
            int(spinbox_bottom.get()),
            int(spinbox_left.get()),
            int(spinbox_right.get())
        )

        for spinbox in (spinbox_top, spinbox_bottom, spinbox_left, spinbox_right):
            spinbox.last_value = int(spinbox.get())

    # Initialize last_value for each spinbox
    for spinbox in (spinbox_top, spinbox_bottom, spinbox_left, spinbox_right):
        spinbox.last_value = 0

    # Function to reset all spinboxes to 0
    def reset_spinboxes():
        for spinbox in (spinbox_top, spinbox_bottom, spinbox_left, spinbox_right):
            spinbox.delete(0, tk.END)
            spinbox.insert(0, "0")
            spinbox.last_value = int(spinbox.get())
        trigger_callback()

    # Attaching the trigger to the spinboxes
    for spinbox in (spinbox_top, spinbox_bottom, spinbox_left, spinbox_right):
        spinbox.config(command=trigger_callback)

    # Attaching the reset function to the button
    button.config(command=reset_spinboxes)

    # Return the frame containing all widgets
    return frame
