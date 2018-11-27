# from Cython.Distutils import build_ext
# from Cython.Build.Dependencies import default_create_extension
import inspect
from distutils.command.build import build
from setuptools.command.install import install

try:
    from setuptools import setup, Extension, Command
except ImportError:
    import ez_setup

    ez_setup.use_setuptools()
    from setuptools import setup, Extension, find_packages


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


setup(
    name="cuckoohash",
    version="0.0.1",
    url='https://github.com/mattpaletta/cuckoohash',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    setup_requires=[],
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
    # ext_modules=[
    #     Extension(name="pycuckoo",
    #               sources=["cuckoo/gpu/pycuckoo.pyx", "cuckoo/gpu/cuckoo.cpp"],
    #               language="c++",
    #               extra_objects=["mtlpp.o"],
    #               extra_compile_args=[],
    #               extra_link_args=[], )],
    cmdclass = {
        # 'build_ext': build_ext,
        'build': BuildCommand,
        'install': InstallCommand
    }
)
