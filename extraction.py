"""Adapted from https://github.com/alliedvision/VimbaPython/blob/master/Examples/asynchronous_grab_opencv.py"""

import threading
import sys
import cv2
import time
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import Optional
from vimba import *


def print_preamble():
    print()
    print("///////////////////////////////////////////////////////")
    print("///          PLD CAMERA IMAGE CAPTURE API           ///")
    print("///////////////////////////////////////////////////////\n")


def print_usage():
    print("Usage:")
    print("    python image_capture.py [moss|rheed]")
    print("    python image_capture.py [/h] [-h]")
    print()


def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + "\n")

    if usage:
        print_usage()

    sys.exit(return_code)


def parse_args() -> Optional[str]:
    args = sys.argv[1:]
    argc = len(args)

    for arg in args:
        if arg in ("/h", "-h"):
            print_usage()
            sys.exit(0)

    if argc > 1:
        abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)

    return None if argc == 0 else args[0]


def get_camera(camera_id: Optional[str]) -> Camera:
    with Vimba.get_instance() as vimba:
        if camera_id:
            try:
                return vimba.get_camera_by_id(camera_id)

            except VimbaCameraError:
                abort("Failed to access Camera '{}'. Abort.".format(camera_id))

        else:
            cams = vimba.get_all_cameras()
            if not cams:
                abort("No Cameras accessible. Abort.")

            return cams[0]


def setup_camera(cam: Camera):
    with cam:
        # Enable auto exposure time setting if camera supports it
        # try:
        #    cam.ExposureAuto.set('Continuous')
        #
        # except (AttributeError, VimbaFeatureError):
        #    pass

        # Set exposure time hard to 10ms
        try:
            cam.ExposureAuto.set("Off")
            cam.ExposureTimeAbs.set(0.01)

        except (AttributeError, VimbaFeatureError):
            pass

        # Enable white balancing if camera supports it
        try:
            cam.BalanceWhiteAuto.set("Continuous")

        except (AttributeError, VimbaFeatureError):
            pass

        # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
        try:
            cam.GVSPAdjustPacketSize.run()

            while not cam.GVSPAdjustPacketSize.is_done():
                pass

        except (AttributeError, VimbaFeatureError):
            pass

        # Query available, open_cv compatible pixel formats
        # prefer color formats over monochrome formats
        cv_fmts = intersect_pixel_formats(cam.get_pixel_formats(), OPENCV_PIXEL_FORMATS)
        color_fmts = intersect_pixel_formats(cv_fmts, COLOR_PIXEL_FORMATS)

        if color_fmts:
            cam.set_pixel_format(color_fmts[0])

        else:
            mono_fmts = intersect_pixel_formats(cv_fmts, MONO_PIXEL_FORMATS)

            if mono_fmts:
                cam.set_pixel_format(mono_fmts[0])

            else:
                abort(
                    "Camera does not support a OpenCV compatible format natively. Abort."
                )


TIME_FORMAT = "%Y%m%d_%H%M%S%f"


class Handler:
    def __init__(self):
        self.shutdown_event = threading.Event()
        timestamp = time.strftime(TIME_FORMAT[:-2])
        self.output_path = Path(timestamp)
        self.output_path.mkdir()

    def __call__(self, cam: Camera, frame: Frame):
        ENTER_KEY_CODE = 13

        key = cv2.waitKey(1)
        if key == ENTER_KEY_CODE:
            self.shutdown_event.set()
            return

        elif frame.get_status() == FrameStatus.Complete:
            timestamp = datetime.now().strftime(TIME_FORMAT)[:-3]
            print("{} acquired {} at {}".format(cam, frame, timestamp), flush=True)

            msg = "Stream from '{}'. Press <Enter> to stop stream."
            cv2.imshow(msg.format(cam.get_name()), frame.as_opencv_image())

            # filename = f"{timestamp}_frame.jpg"
            # cv2.imwrite(filename, frame.as_opencv_image())

            data = frame.as_numpy_ndarray()
            filepath = self.output_path / f"{timestamp}_frame.npy"
            np.save(filepath, data.reshape(data.shape[:-1]))

        cam.queue_frame(frame)


CAMERA_IDS = {
    "moss": "DEV_000F3102C6CA",
    "rheed": "DEV_000F314F7A84",
}


def main():
    print_preamble()
    cam = parse_args()

    if not (cam_id := CAMERA_IDS.get(cam)):
        print(f"Available cameras: 'moss', 'rheed'; got {cam}")
        return

    with Vimba.get_instance():
        with get_camera(cam_id) as cam:
            # Start Streaming, wait for five seconds, stop streaming
            setup_camera(cam)
            handler = Handler()

            try:
                # Start Streaming with a custom buffer of 10 Frames (defaults to 5)
                cam.start_streaming(handler=handler, buffer_count=10)
                handler.shutdown_event.wait()

            finally:
                cam.stop_streaming()


if __name__ == "__main__":
    main()
