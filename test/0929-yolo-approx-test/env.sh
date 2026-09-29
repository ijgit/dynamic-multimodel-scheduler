# Usage: source /workspace/toyota/test/0929-yolo-gemm-dag/env.sh
# Builds on the 0928 environment: CUDA 12.0 nvcc for the megakernel, mirage venv ($PY), GPU 6 unless MPK_GPU is set.
source /workspace/toyota/dynamic-multimodel-scheduler/test/0929-yolo-approx-test/env.sh
export T0929G=/workspace/toyota/test/0929-yolo-gemm-dag
export PYTHONPATH=$T0929G/common${PYTHONPATH:+:$PYTHONPATH}
# cuBLAS from the pip wheels must be found when the tests run under nsys.
_NV=$($PY -c "import nvidia, os; print(os.path.dirname(nvidia.__path__[0]))" 2>/dev/null)/nvidia
export LD_LIBRARY_PATH=$_NV/cublas/lib:$LD_LIBRARY_PATH
unset _NV
