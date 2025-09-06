# -*- coding: utf-8 -*-
"""The setup.py for unirobot."""
from pkg_resources import parse_requirements
from setuptools import find_packages
from setuptools import setup

from slot_entry import unirobot_slot_entry  # noqa: I900


# Parse content from `README.md` as long description.
with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

# Parse content from `requirements.txt` as install requires.
with open("requirements.txt", encoding="utf-8") as fh:
    install_requires = [str(requirement) for requirement in parse_requirements(fh)]

setup(
    author="matrix97317",
    author_email="494649824@qq.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
    ],
    description="UniRobot is an embodied intelligent software framework that integrates the robot brain (data, models, model training) with the robot body (perception, model inference, control).",
    entry_points=unirobot_slot_entry,
    install_requires=install_requires,
    license="Apache License 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    name="unirobot",
    packages=find_packages(exclude=["dist.*", "dist", "tests.*", "tests"]),
    python_requires=">=3.8",
    url="https://github.com/matrix97317/UniRobot.git",
)
