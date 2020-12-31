import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="embarrassingly",
    version="0.0.6",
    description="Pimp your objective function for faster, robust optimization",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/embarrassingly",
    author="microprediction",
    author_email="pcotton@intechinvestments.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["embarrassingly"],
    test_suite='pytest',
    tests_require=['pytest'],
    include_package_data=True,
    install_requires=["wheel","pathlib","numpy","scipy","scikit-learn","smt","tdigest","numpy","optuna","deap","microprediction","pandas"],
    entry_points={
        "console_scripts": [
            "embarrassingly=embarrassingly.__main__:main",
        ]
    },
)
