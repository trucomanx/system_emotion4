#!/usr/bin/python

from setuptools import setup, find_packages
import os 

def func_read(fname):
    return open(os.path.join(os.path.dirname(__file__),fname)).read();

setup(
    name   ='SystemEmotion4Lib',
    version='0.1.0',
    author='Fernando Pujaico Rivera',
    author_email='fernando.pujaico.rivera@gmail.com',
    maintainer='Fernando Pujaico Rivera',
    maintainer_email='fernando.pujaico.rivera@gmail.com',
    #scripts=['bin/script1','bin/script2'],
    url='https://github.com/trucomanx/system_emotion4',
    license='GPLv3',
    description='Functions to detect 4 emotions',
    #long_description=func_read('README.md'),
    include_package_data=True,
    packages=['SystemEmotion4Lib'], #find_packages(where="SystemEmotion4Lib"),
    package_dir={'SystemEmotion4Lib':'SystemEmotion4Lib'},
    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        #'': ['*.py','*.h5'],
        'SystemEmotion4Lib': ['*.py','models/*.h5']
    },
    install_requires=[ #"Django >= 1.1.1",
       "OpenPifPafTools",
       "FaceEmotion4Lib",
       "BodyEmotion4Lib",
       "SkeletonEmotion4Lib",
       "FusionEmotion4Lib",
       "tensorflow"
    ],
)

#! python setup.py sdist bdist_wheel
# Upload to PyPi
# or 
#! pip3 install dist/SystemEmotion4Lib-0.1.0.tar.gz 
