import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auto_peak_labeler",
    version="23.08.03",
    author="Cong Wang",
    author_email="wangimagine@gmail.com",
    description="Auto Bragg peak labeler with profile fitting.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carbonscott/auto-peak-labeler",
    keywords = ['SFX', 'X-ray', 'Model Fitting', 'LCLS'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
