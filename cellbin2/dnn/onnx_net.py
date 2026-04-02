import site
from os import path

import onnxruntime

from cellbin2.dnn import BaseNet
from cellbin2.utils import clog


class OnnxNet(BaseNet):
    _CUDA_RUNTIME_LIB_DIR_NAMES = (
        'cudnn',
        'cublas',
        'cuda_runtime',
        'cuda_nvrtc',
        'cufft',
        'curand',
        'cusolver',
        'cusparse',
        'nvjitlink',
        'nccl',
        'nvtx',
        'cuda_cupti',
    )

    def __init__(self, model_path, gpu="-1", num_threads=0):
        super(OnnxNet, self).__init__()
        self._providers = ['CPUExecutionProvider']
        self._providers_id = [{'device_id': "-1"}]
        self._model = None
        self._gpu = int(gpu)
        self._model_path = model_path
        self._input_name = 'input_1'
        self._output_name = None
        self._num_threads = num_threads
        self._input_shape = (0, 0, 0)
        self._f_init()

    @staticmethod
    def _f_cpu_providers():
        return ['CPUExecutionProvider'], [{'device_id': '-1'}]

    def _f_gpu_providers(self):
        return ['CUDAExecutionProvider', 'CPUExecutionProvider'], [{'device_id': str(self._gpu)}, {}]

    @classmethod
    def _f_nvidia_lib_dirs(cls):
        lib_dirs = []
        seen = set()
        for site_pkg in site.getsitepackages() + [site.getusersitepackages()]:
            nvidia_root = path.join(site_pkg, 'nvidia')
            for lib_name in cls._CUDA_RUNTIME_LIB_DIR_NAMES:
                lib_dir = path.join(nvidia_root, lib_name, 'lib')
                if path.isdir(lib_dir) and lib_dir not in seen:
                    lib_dirs.append(lib_dir)
                    seen.add(lib_dir)
        return lib_dirs

    @classmethod
    def _f_cuda_runtime_hint(cls):
        msg = (
            "CUDAExecutionProvider init failed. Verify that onnxruntime-gpu and the CUDA/cuDNN "
            f"runtime versions are compatible. Current ORT version is {onnxruntime.__version__}. "
            "For ORT 1.19.x, CUDA EP requires CUDA 12.* and cuDNN 9.*."
        )
        lib_dirs = cls._f_nvidia_lib_dirs()
        if lib_dirs:
            export_paths = ':'.join(lib_dirs)
            msg = (
                f"{msg} Detected NVIDIA runtime libraries under site-packages. "
                "Export their lib directories before launching cellbin2, for example: "
                f"LD_LIBRARY_PATH={export_paths}:${{LD_LIBRARY_PATH}}"
            )
        return msg

    def _f_init(self):
        if self._gpu > -1:
            available_providers = onnxruntime.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                self._providers, self._providers_id = self._f_gpu_providers()
            else:
                clog.warning(
                    "CUDAExecutionProvider is unavailable in the current ONNX Runtime build. "
                    f"Available providers: {available_providers}. Falling back to CPU."
                )
                self._providers, self._providers_id = self._f_cpu_providers()
        self._f_load_model()

    def _f_load_model(self):
        if path.exists(self._model_path):
            clog.info(f"loading weight from {self._model_path}")
            sessionOptions = onnxruntime.SessionOptions()
            try:
                if (self._gpu < 0) and (self._num_threads > 0):
                    sessionOptions.intra_op_num_threads = self._num_threads
                self._model = onnxruntime.InferenceSession(self._model_path, providers=self._providers,
                                                           provider_options=self._providers_id,
                                                           sess_options=sessionOptions)
            except Exception as exc:
                if self._gpu > -1:
                    clog.warning(
                        f"Failed to initialize ONNX Runtime with providers {self._providers}: {exc}"
                    )
                    clog.warning(self._f_cuda_runtime_hint())
                    clog.warning(
                        f"ORT available providers: {onnxruntime.get_available_providers()}"
                    )
                if self._num_threads > 0:
                    sessionOptions.intra_op_num_threads = self._num_threads
                self._providers, self._providers_id = self._f_cpu_providers()
                self._model = onnxruntime.InferenceSession(self._model_path, providers=self._providers,
                                                           provider_options=self._providers_id,
                                                           sess_options=sessionOptions)
                clog.info(f"Warning!!! GPU call failed, onnx work on cpu,threads {self._num_threads}")

            active_provider = self._model.get_providers()[0]
            expected_provider = self._providers[0]
            if active_provider == expected_provider:
                if active_provider == 'CPUExecutionProvider':
                    clog.info(f"onnx work on cpu,threads {self._num_threads}")
                else:
                    clog.info(f"onnx work on gpu {self._gpu}")
            else:
                clog.warning(f'Warning!!! expected: {expected_provider}, active: {active_provider}')
                if active_provider == 'CPUExecutionProvider':
                    if expected_provider == 'CUDAExecutionProvider':
                        clog.warning(self._f_cuda_runtime_hint())
                        clog.warning(
                            f"ORT build available providers: {onnxruntime.get_available_providers()}"
                        )
                    clog.info(f'Warning!!! GPU call failed, onnx work on cpu,threads {self._num_threads}')
                if active_provider == 'CUDAExecutionProvider':
                    clog.info(f'onnx work on gpu')
            self._input_name = self._model.get_inputs()[0].name
            self._input_shape = tuple(self._model.get_inputs()[0].shape[1:])
            self._output_shape = tuple(self._model.get_outputs()[0].shape)
        else:
            raise Exception(f"Weight path '{self._model_path}' does not exist")

    def f_predict(self, data):
        pred = self._model.run(self._output_name, {self._input_name: data})
        return pred

    def f_get_input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape
