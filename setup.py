from setuptools import setup

setup(name='lmpio',
      version='0.0.2',
      description='A module to read and write LAMMPS input and output files',
      url='https://github.com/eskoruppa/lmpio',
      author='Enrico Skoruppa',
      author_email='enrico.skoruppa@gmail.com',
      license='GNU General Public License v2.0',
      packages=['lmpio'],
      package_dir={
            'lmpio': 'lmpio',
      },
      install_requires=[
          'numpy>=1.13.3',
      ]
      ) 
