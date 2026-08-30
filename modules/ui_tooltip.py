"""Lightweight hover tooltip for CustomTkinter widgets."""

from tkinter import TclError

import customtkinter as ctk


class ToolTip:
    """Show a floating tooltip popup when the user hovers over a widget.

    Usage:
        ToolTip(my_button, "Helpful description text")
    """

    def __init__(self, widget: ctk.CTkBaseClass, text: str, delay: int = 500):
        self._widget = widget
        self._text = text
        self._delay = delay
        self._tooltip_window = None
        self._after_id = None

        widget.bind("<Enter>", self._schedule_show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<Destroy>", self._on_destroy, add="+")

    def _schedule_show(self, event=None):
        self._cancel()
        self._after_id = self._widget.after(self._delay, self._show)

    def _show(self):
        if self._tooltip_window is not None:
            return

        x = self._widget.winfo_rootx() + 20
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 5

        self._tooltip_window = tw = ctk.CTkToplevel(self._widget)
        tw.withdraw()
        tw.overrideredirect(True)

        label = ctk.CTkLabel(
            tw,
            text=self._text,
            fg_color="#333333",
            text_color="#EEEEEE",
            corner_radius=6,
            padx=8,
            pady=4,
        )
        label.pack()

        tw.update_idletasks()

        # Clamp to screen bounds
        screen_w = tw.winfo_screenwidth()
        screen_h = tw.winfo_screenheight()
        tip_w = tw.winfo_reqwidth()
        tip_h = tw.winfo_reqheight()

        if x + tip_w > screen_w:
            x = screen_w - tip_w - 5
        if y + tip_h > screen_h:
            y = self._widget.winfo_rooty() - tip_h - 5

        tw.geometry(f"+{x}+{y}")
        tw.deiconify()

    def _hide(self, event=None):
        self._cancel()
        if self._tooltip_window is not None:
            try:
                self._tooltip_window.destroy()
            except TclError:
                pass
            self._tooltip_window = None

    def _on_destroy(self, event=None):
        """Release callbacks and windows when the owner widget is destroyed."""
        self._hide()

    def _cancel(self):
        if self._after_id is not None:
            try:
                self._widget.after_cancel(self._after_id)
            except TclError:
                pass
            finally:
                self._after_id = None
