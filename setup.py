from setuptools import find_packages, setup

setup(
    name="airline_analysis",
    version="1.0.0",
    description="Airline performance and trend analysis visualizations",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    author="Your Name",
    author_email="youremail@example.com",
    license="MIT",
    packages=find_packages(),
    install_requires=["pandas", "plotly", "openpyxl"],
    entry_points={
        "console_scripts": [
            "run-analysis=sample.core:run_analysis",
        ],
    },
    python_requires=">=3.7",
)
