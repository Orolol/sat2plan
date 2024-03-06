from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='sat2plan',
      version="0.0.1",
      description="sat2plan Model",
      license="MIT",
      author="Gaétan Martin",
      author_email="gaetan@0r0.fr",
      url="https://github.com/orolol/sat2plan",
      install_requires=requirements,
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
