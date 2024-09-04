# Problem



    qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/USERNAME/.local/lib/python3.10/site-packages/cv2/qt/plugins" even though it was found.
    This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

    Available platform plugins are: xcb, eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx.

    Aborted (core dumped)

We have 2 solutions:

    cd ~/.local/lib/python3.10/site-packages/cv2/qt/plugins
    mv platforms platforms_old
    ln -s /usr/lib/x86_64-linux-gnu/qt5/plugins/platforms

or

    export DISPLAY=':0.0'
