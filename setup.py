import sys
import os
from Cython.Distutils import build_ext
try:
    from setuptools import setup, find_packages
    from setuptools.extension import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
prjdir = os.path.dirname(__file__)

import numpy

def read(filename):
    return open(os.path.join(prjdir, filename)).read()

extra_compile_args = ['-fopenmp','-std=c++11', '-Wcpp']
extra_link_args = ['-fopenmp']

if sys.platform == 'darwin':
    # On macOS, we need to specify the path to LLVM libraries and headers
    # if installed via Homebrew, otherwise it will use the system's LLVM.
    # This is necessary for OpenMP support.
    if not os.path.exists('/opt/homebrew/opt/llvm/lib'):
        raise RuntimeError("LLVM not found. Please install LLVM via Homebrew: `brew install llvm`")
    # Set the environment variables to use Homebrew's LLVM
    os.environ['CC'] = '/opt/homebrew/opt/llvm/bin/clang'
    os.environ['CXX'] = '/opt/homebrew/opt/llvm/bin/clang++'

    extra_compile_args += ["-L/opt/homebrew/opt/llvm/lib", "-I/opt/homebrew/opt/llvm/include"]


libraries = []
library_dirs = []
include_dirs = []
exec(open('version.py').read())
setup(
    name='eif',
    version=__version__,
    author='Matias Carrasco Kind , Sahand Hariri, Seng Keat Yeoh',
    author_email='mcarras2@illinois.edu',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("eif",
                 sources=["_eif.pyx", "eif.cxx"],
                 include_dirs=[numpy.get_include()],
                 extra_compile_args=extra_compile_args,
                 extra_link_args=extra_link_args,
                 language="c++")],
    scripts=[],
    py_modules=['version'],
    packages=[],
    license='License.txt',
    include_package_data=True,
    description='Extended Isolation Forest for anomaly detection',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/sahandha/eif',
    install_requires=["numpy", "cython", "scikit-learn"],
    zip_safe=False,
)