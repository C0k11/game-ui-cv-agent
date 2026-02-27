import win32gui
from typing import Optional, Tuple

class GameWindow:
    """
    Utility class to find the Steam 'Blue Archive' window and get its bounding box.
    This is used to constrain DXcam capture to just the game region.
    """
    def __init__(self, window_title: str = "Blue Archive"):
        self.window_title = window_title
        self.hwnd = None

    def _find_window_callback(self, hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if self.window_title in title:
                self.hwnd = hwnd

    def find_window(self) -> bool:
        """Find the game window by title."""
        self.hwnd = None
        win32gui.EnumWindows(self._find_window_callback, None)
        return self.hwnd is not None

    def get_region(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the window region (left, top, right, bottom).
        Returns None if window is not found.
        """
        if not self.hwnd:
            if not self.find_window():
                return None
                
        # Get client rect (excludes window borders/titlebar)
        try:
            rect = win32gui.GetClientRect(self.hwnd)
            # Convert client coordinates to screen coordinates
            left_top = win32gui.ClientToScreen(self.hwnd, (rect[0], rect[1]))
            right_bottom = win32gui.ClientToScreen(self.hwnd, (rect[2], rect[3]))
            
            # left, top, right, bottom
            return (left_top[0], left_top[1], right_bottom[0], right_bottom[1])
        except Exception as e:
            print(f"[GameWindow] Error getting region: {e}")
            return None
