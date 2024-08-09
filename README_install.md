# Packaging

Download the source code
    
    git clone https://github.com/trucomanx/fcnn_emotion4_fusion

The next command generates the `dist/FusionEmotion4Lib-VERSION.tar.gz` file.

    cd fcnn_emotion4_fusion/library
    python3 setup.py sdist

For more informations use `python setup.py --help-commands`

# Install 

Install the packaged library

    pip3 install dist/FusionEmotion4Lib-*.tar.gz

# Uninstall

    pip3 uninstall FusionEmotion4Lib
