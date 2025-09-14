# -*- coding: utf-8 -*-
"""OpenCV Camera Sensor."""
import logging
import math
import time
from threading import Event, Lock, Thread

import cv2
import numpy as np

from unirobot.robot.device_interface import BaseDevice


logger = logging.getLogger(__name__)


class OpenCVCamera(BaseDevice):
    """The abstract interface of Robot Device.

    Args:
        host_name (str): Device's ID, such as IP.
        port (str) : Device' Port, such as IP's port, UART prot.
    """

    ROTATE = ["0", "90", "90c", "180"]
    COLOR_MODE = ["BGR", "RGB"]
    CV2_ROTATE = {
        "0": None,
        "90": cv2.ROTATE_90_CLOCKWISE,
        "90c": cv2.ROTATE_90_COUNTERCLOCKWISE,
        "180": cv2.ROTATE_180,
    }

    def __init__(
        self,
        host_name: str = "localhost",
        port: str = "1234",
        fps: int = 25,
        width: int = 640,
        height=480,
        color_mode="BGR",
        warmup: bool = True,
        rotate: str = "0",
    ):
        """Init."""
        # self._host_name = host_name
        # self._port = port
        super().__init__(host_name=host_name, port=port)
        if rotate not in self.ROTATE:
            raise ValueError(
                f"Current rotate degree {rotate} not be supported. {self.ROTATE}"
            )
        if color_mode not in self.COLOR_MODE:
            raise ValueError(
                f"Current color mode {color_mode} not be supported. {self.COLOR_MODE}"
            )

        self.fps = fps
        self.width = width
        self.height = height
        self.color_mode = color_mode
        self.warmup = warmup

        self.is_connected = False

        self.videocapture = None
        self.fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")

        self.thread = None
        self.stop_event = None
        self.frame_lock = Lock()
        self.latest_frame = None
        self.new_frame_event = Event()

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if rotate in ["90"]:
                self.capture_width, self.capture_height = self.height, self.width
        self.rotation = self.CV2_ROTATE[rotate]

    def __str__(self) -> str:
        """Get device string."""
        return f"{self._host_name}({self._port})"

    def open(self, *args, **kwargs) -> None:
        """Open a device."""
        # Use 1 thread for OpenCV operations to avoid potential conflicts or
        # blocking in multi-threaded applications, especially during data collection.
        cv2.setNumThreads(1)

        self.videocapture = cv2.VideoCapture(self._port)

        if not self.videocapture.isOpened():
            self.videocapture.release()
            self.videocapture = None
            raise RuntimeError(
                f"Failed to open {self}."
                f"Run `pipenv run unirobot find-camera` to find available cameras."
            )
        self.is_connected = self.videocapture.isOpened()
        self.configure()

        if self.warmup:
            start_time = time.time()
            while time.time() - start_time < 1:
                self.get()
                time.sleep(0.1)

        logger.info(f"{self} connected.")

    def configure(self, *args, **kwargs) -> None:
        """Configure a device."""
        if not self.is_connected:
            raise RuntimeError(
                f"Cannot configure settings for {self} as it is not connected."
            )

        default_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        default_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if self.width is None or self.height is None:
            self.width, self.height = default_width, default_height
            self.capture_width, self.capture_height = default_width, default_height
            if self.rotation in [
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            ]:
                self.width, self.height = default_height, default_width
                self.capture_width, self.capture_height = default_width, default_height
        else:
            width_success = self.videocapture.set(
                cv2.CAP_PROP_FRAME_WIDTH, float(self.capture_width)
            )
            height_success = self.videocapture.set(
                cv2.CAP_PROP_FRAME_HEIGHT, float(self.capture_height)
            )

            actual_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            if not width_success or self.capture_width != actual_width:
                raise RuntimeError(
                    f"{self} failed to set capture_width={self.capture_width} ({actual_width=}, {width_success=})."
                )

            actual_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if not height_success or self.capture_height != actual_height:
                raise RuntimeError(
                    f"{self} failed to set capture_height={self.capture_height} ({actual_height=}, {height_success=})."
                )

        fourcc_succ = self.videocapture.set(cv2.CAP_PROP_FOURCC, self.fourcc)
        actual_fourcc = self.videocapture.get(cv2.CAP_PROP_FOURCC)
        if not fourcc_succ or actual_fourcc != self.fourcc:
            raise RuntimeError(
                f"{self} failed to set fourcc={self.fourcc} ({actual_fourcc=}, {fourcc_succ=})."
            )

        if self.fps is None:
            self.fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        else:
            success = self.videocapture.set(cv2.CAP_PROP_FPS, float(self.fps))
            actual_fps = self.videocapture.get(cv2.CAP_PROP_FPS)
            # Use math.isclose for robust float comparison
            if not success or not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
                raise RuntimeError(
                    f"{self} failed to set fps={self.fps} ({actual_fps=})."
                )

    def _post_process(self, img: np.ndarray) -> np.ndarray:
        """Post process video frame."""
        requested_color_mode = self.color_mode

        if requested_color_mode not in self.COLOR_MODE:
            raise ValueError(
                f"Invalid color mode '{requested_color_mode}'. {self.COLOR_MODE}."
            )

        h, w, c = img.shape

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        if c != 3:
            raise RuntimeError(
                f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR)."
            )

        processed_image = img
        if requested_color_mode == "RGB":
            processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.rotation in [
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def get(self, *args, **kwargs) -> None:
        """Get info from device."""
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")

        start_time = time.perf_counter()

        ret, frame = self.videocapture.read()

        if not ret or frame is None:
            raise RuntimeError(f"{self} read failed (status={ret}).")

        processed_frame = self._post_process(frame)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return processed_frame

    def _get_loop(self):
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads a color frame
        2. Stores result in latest_frame (thread-safe)
        3. Sets new_frame_event to notify listeners

        Stops on DeviceNotConnectedError, logs other errors and continues.
        """
        while not self.stop_event.is_set():
            try:
                color_image = self.get()

                with self.frame_lock:
                    self.latest_frame = color_image
                self.new_frame_event.set()

            except RuntimeError:
                break
            except Exception as e:
                logger.warning(
                    f"Error reading frame in background thread for {self}: {e}"
                )

    def _start_get_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._get_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_get_thread(self) -> None:
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def get_async(self, timeout_ms: float = 200) -> np.ndarray:
        """
        Reads the latest available frame asynchronously.

        This method retrieves the most recent frame captured by the background
        read thread. It does not block waiting for the camera hardware directly,
        but may wait up to timeout_ms for the background thread to provide a frame.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 200ms (0.2 seconds).

        Returns:
            np.ndarray: The latest captured frame as a NumPy array in the format
                       (height, width, channels), processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame becomes available within the specified timeout.
            RuntimeError: If an unexpected error occurs.
        """
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_get_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(
                f"Internal error: Event set but no frame available for {self}."
            )

        return frame

    def put(self, *args, **kwargs) -> None:
        """Put info to device."""
        raise NotImplementedError()

    def close(self, *args, **kwargs) -> None:
        """Close a device."""
        if not self.is_connected and self.thread is None:
            raise RuntimeError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_get_thread()

        if self.videocapture is not None:
            self.videocapture.release()
            self.videocapture = None

        logger.info(f"{self} disconnected.")
