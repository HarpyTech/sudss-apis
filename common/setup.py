from setuptools import setup, find_packages

setup(
    name="common-utils",
    version="0.1.0",
    description="Shared utilities for the monorepo (common_utils)",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        # put common runtime dependencies here if any
    ],
)
