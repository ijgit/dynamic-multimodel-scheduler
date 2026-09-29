# Usage: source env.sh   (from any directory; paths are taken from where this file is)
# CUDA 12.0 nvcc for the MPK megakernel, the mirage venv ($PY), GPU 6 unless MPK_GPU is set.
T0929G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export T0929G
export MIRAGE_HOME=${MIRAGE_HOME:-/workspace/toyota/mirage}
# The megakernel needs nvcc >= 12 (/usr/local/cuda is 11.8 on this machine).
if [ -z "${MPK_CUDA_HOME:-}" ]; then
  if [ -x /opt/cuda-12.0-mpk/bin/nvcc ]; then MPK_CUDA_HOME=/opt/cuda-12.0-mpk; else MPK_CUDA_HOME=/usr/local/cuda-12.0; fi
fi
export MPK_CUDA_HOME
export PATH=$MPK_CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPK_CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_VISIBLE_DEVICES=${MPK_GPU:-6}
export PYTHONPATH=$T0929G/common:$MIRAGE_HOME/python${PYTHONPATH:+:$PYTHONPATH}
export PY=$MIRAGE_HOME/venv/bin/python
# MPK's op-level AnnotatedGraph (layers, fork/join, topo order) goes to 02_mpk/out/<tag>/compile.log.
export MIRAGE_DUMP_ANNOTATED_GRAPH=1
# Shells without a UTF-8 locale make open() default to ASCII and MPK's task graph dump fails.
export PYTHONUTF8=1
# cuBLAS from the pip wheels must be found when the tests run under nsys.
_NV=$($PY -c "import nvidia, os; print(os.path.dirname(nvidia.__path__[0]))" 2>/dev/null)/nvidia
export LD_LIBRARY_PATH=$_NV/cublas/lib:$LD_LIBRARY_PATH
unset _NV
