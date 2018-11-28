from Cython.Build import cythonize

import inspect
from distutils.command.build import build

from coriander.coriander import cu_to_cl_bin

try:
    from setuptools import Command, find_packages
except ImportError:
    import ez_setup

    ez_setup.use_setuptools()
    from setuptools import find_packages


from distutils.command.install import install
from distutils.core import setup
from distutils.extension import Extension


class BuildCommand(build):
    def run(self):
        build.run(self)


class InstallCommand(install):
    def run(self):
        if not self._called_from_setup(inspect.currentframe()):
            # Run in backward-compatibility mode to support bdist_* commands.
            install.run(self)
        else:
            install.do_egg_install(self)  # OR: install.do_egg_install(self)


# class BuildExt(build_ext):
#     def run(self):
#         print("Running build ext")
#         build_ext.run(self)


extensions = Extension("pycuckoo",
                  ["cuckoo/gpu/pycuckoo.pyx"],
                  language="c++",
                  libraries=["cuckoo"],
                  library_dirs = ["cuckoo/gpu/lib"],
                  include_dirs = ["cuckoo/gpu/lib"],
                  extra_compile_args=["-mmacosx-version-min=10.13", "-fPIC"],
                  extra_link_args=["-mmacosx-version-min=10.13", "-fPIC"], )

setup(
    name="cuckoohash",
    version="0.0.1",
    url='https://github.com/mattpaletta/cuckoohash',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    setup_requires=["pycoriander"],
    author="Matthew Paletta",
    author_email="mattpaletta@gmail.com",
    description="Cuckoo Hash GPU/CPU implementation",
    license="BSD",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Communications',
    ],
    ext_modules=cythonize([extensions]),
    cmdclass = {
        # 'build_ext': build_ext,
        'build': BuildCommand,
        'install': InstallCommand
    }
)
