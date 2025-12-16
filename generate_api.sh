dials_python=/home/marko/stfc/dials_build/dials

sphinx-apidoc -f -o docs/api dynamic --module-first
rm docs/api/modules.rst  # Remove the docs wrapper (we have our own)
source ${dials_python} 
cd docs
make clean
make html
make html
cd ..
conda deactivate


