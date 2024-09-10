# Problem 1

## Error message

    qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/USERNAME/.local/lib/python3.10/site-packages/cv2/qt/plugins" even though it was found.
    This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

    Available platform plugins are: xcb, eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx.

    Aborted (core dumped)
## Solutions
We have 2 solutions:

    cd ~/.local/lib/python3.10/site-packages/cv2/qt/plugins
    mv platforms platforms_old
    ln -s /usr/lib/x86_64-linux-gnu/qt5/plugins/platforms

or

    export DISPLAY=':0.0'

# Problem 2

## Error message

    2024-09-10 18:01:31.221710: E tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:425] Loaded runtime CuDNN library: 8.5.0 but source was compiled with: 8.6.0.  CuDNN library needs to have matching major version and equal or higher minor version. If using a binary install, upgrade your CuDNN library.  If building from sources, make sure the library loaded at runtime is compatible with the version specified during compile configuration.

## Solutions
Update the `nvidia-cudnn-cu11`, because some version call CuDNN library: 8.5.0

    pip install -U nvidia-cudnn-cu11
