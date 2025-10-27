# lib/utils/tensorrt_utils.py
import torch
#TODO: install the compiled ONNX models into the src/controllers/resource/models folder
try:
    import tensorrt as trt
    import numpy as np
except ImportError:
    print("ERROR: TensorRT or numpy not found. TensorRT utilities require these packages.")
    # Allow import to proceed for type definitions, but functionality will be disabled.
    trt = None
    np = None

# Import ONNX for validation
try:
    import onnx
except ImportError:
    onnx = None
    print("WARNING: 'onnx' package not found. Skipping ONNX validation. (pip install onnx)")

import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize TensorRT logger
if trt:
    # Set severity level to VERBOSE for detailed debugging, INFO for normal operation.
    # Set severity level to INFO
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    logger.info(f"Initialized with TensorRT Version: {trt.__version__}")
else:
    TRT_LOGGER = None

class TensorRTEngineWrapper:
    """
    Wraps a TensorRT engine for inference using PyTorch tensors (zero-copy).
    Fully compatible with the modern TensorRT 10.x API (V3 execution).
    """
    def __init__(self, engine_path, device):
        if trt is None:
            raise ImportError("TensorRT is not available.")
        self.engine_path = engine_path
        self.device = device
        
        # Initialize attributes
        self.engine = None
        self.context = None
        self.stream = None
        self.inputs_meta = []
        self.outputs_meta = []

        self.engine = self.load_engine(engine_path)
        
        if self.engine is not None:
            # Engine loaded successfully, initialize context and metadata
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Failed to create TensorRT execution context.")

            # Use the current PyTorch CUDA stream for asynchronous execution
            self.stream = torch.cuda.current_stream(device=device).cuda_stream
            # Get metadata using the modern API
            self.inputs_meta, self.outputs_meta = self.get_tensors_meta()
        
        elif os.path.exists(engine_path):
             # Engine is None but file exists -> Deserialization failed (e.g., corruption or incompatibility)
             # Raise error so the caller knows loading failed and can proceed to rebuild.
             raise RuntimeError(f"Failed to deserialize TensorRT engine from {engine_path}. Engine might be corrupted or incompatible.")
        # Else: Engine is None and file does not exist -> Expected behavior, caller will build.

    def load_engine(self, engine_path):
        """Load a serialized TensorRT engine."""
        if not os.path.exists(engine_path):
            # This is expected if the engine hasn't been built yet.
            return None
        logger.info(f"Loading TensorRT engine from: {engine_path}")
        
        # Load the TensorRT runtime
        runtime = trt.Runtime(TRT_LOGGER)
        if runtime is None:
            raise RuntimeError("Failed to initialize TensorRT Runtime.")

        try:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
                engine = runtime.deserialize_cuda_engine(engine_data)
                return engine
        except Exception as e:
            logger.error(f"Error during engine deserialization: {e}")
            return None

    # Modernized metadata retrieval (Replaces get_bindings_meta)
    def get_tensors_meta(self):
        """
        Retrieve metadata (name, shape definition, dtype) using the TensorRT 10.x API.
        """
        inputs = []
        outputs = []
        # Iterate using num_io_tensors (Modern API)
        for i in range(self.engine.num_io_tensors):
            # Access properties by name.
            name = self.engine.get_tensor_name(i)
            
            # Get shape definition (may contain -1 for dynamic dimensions)
            shape_definition = self.engine.get_tensor_shape(name)
            
            # Map TRT dtype to PyTorch dtype
            dtype_trt = self.engine.get_tensor_dtype(name)
            torch_dtype = self._map_trt_dtype_to_torch(dtype_trt)

            # We rely on 'name' in the modern API (V3), not fixed indices.
            info = {'name': name, 'shape_definition': shape_definition, 'dtype': torch_dtype}

            # Check I/O mode (Modern API)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(info)
            else:
                outputs.append(info)
        return inputs, outputs

    def _map_trt_dtype_to_torch(self, trt_dtype):
        # Optimized mapping for common types
        if trt_dtype == trt.DataType.FLOAT:
            return torch.float32
        elif trt_dtype == trt.DataType.HALF:
            return torch.float16
        # Fallback using numpy bridge
        try:
            np_dtype = trt.nptype(trt_dtype)
            return torch.from_numpy(np.empty(0, dtype=np_dtype)).dtype
        except TypeError:
            raise TypeError(f"Unsupported TensorRT data type: {trt_dtype}")

    # Modernized execution flow (V3)
    def __call__(self, *input_tensors):
        """
        Execute inference asynchronously using the modern TensorRT 10.x API (execute_async_v3).
        """
        if self.context is None:
             raise RuntimeError("TensorRT execution context is not initialized.")
             
        if len(input_tensors) != len(self.inputs_meta):
             raise ValueError(f"Expected {len(self.inputs_meta)} inputs, but got {len(input_tensors)}.")

        # 1. Prepare inputs, convert types, set dynamic shapes, and set addresses (Zero-copy)
        # In TRT 10.x, we use set_tensor_address and set_input_shape by name.
        
        for i, input_meta in enumerate(self.inputs_meta):
            input_tensor = input_tensors[i]
            tensor_name = input_meta['name']

            if not input_tensor.is_cuda:
                 raise RuntimeError("Input tensors must be on CUDA device.")

            # Handle Data Type Conversion (Crucial for FP16 engines)
            if input_tensor.dtype != input_meta['dtype']:
                # If engine expects FP16 and input is FP32, convert it.
                input_tensor = input_tensor.to(input_meta['dtype'])
            
            # Set the input shape if it's dynamic (Modern API: set_input_shape)
            # We check if the currently set shape differs before attempting to change it (optimization)
            if self.context.get_tensor_shape(tensor_name) != input_tensor.shape:
                if not self.context.set_input_shape(tensor_name, input_tensor.shape):
                    raise RuntimeError(f"Failed to set dynamic input shape for {tensor_name} to {input_tensor.shape}. Check optimization profiles.")

            # Set the device pointer (Modern API: set_tensor_address)
            self.context.set_tensor_address(tensor_name, input_tensor.data_ptr())

        # 2. Prepare output buffers (Zero-copy)
        output_tensors = []
        for output_meta in self.outputs_meta:
            tensor_name = output_meta['name']
            
            # Get the actual output shape from the context (handles dynamic shapes)
            output_shape = self.context.get_tensor_shape(tensor_name)

            # Allocate output tensor on GPU
            output_tensor = torch.empty(tuple(output_shape), dtype=output_meta['dtype'], device=self.device)
            output_tensors.append(output_tensor)
            
            # Set the device pointer for the output buffer
            self.context.set_tensor_address(tensor_name, output_tensor.data_ptr())

        # 3. Execute inference asynchronously (Modern API: execute_async_v3)
        # This version does not use the 'bindings' array.
        if not self.context.execute_async_v3(stream_handle=self.stream):
            raise RuntimeError("TensorRT inference execution (V3) failed.")

        # Note on Synchronization: PyTorch automatically manages CUDA streams.
        
        # Mimic PyTorch behavior: return single tensor if only one output
        if len(output_tensors) == 1:
            return output_tensors[0]
        return output_tensors

# Helper function for ONNX conversion (No changes needed here)
def convert_pytorch_to_onnx(model, dummy_inputs, onnx_path, input_names, output_names):
    """Converts a PyTorch model to ONNX format with dynamic batch size."""
    logger.info(f"Converting PyTorch model to ONNX: {onnx_path}")
    # Ensure model is in evaluation mode
    model.eval()

    # Define dynamic axes: the first dimension (index 0) is the 'batch_size'
    dynamic_axes = {name: {0: 'batch_size'} for name in input_names + output_names}
    
    # Ensure dummy inputs are FP32 for tracing, even if the target engine is FP16.
    dummy_inputs_fp32 = tuple(t.float() for t in dummy_inputs)

    torch.onnx.export(
        model,
        dummy_inputs_fp32,
        onnx_path,
        export_params=True,
        opset_version=13, # Opset 13 is generally well supported
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    # --- ONNX Validation ---
    logger.info("ONNX conversion finished. Starting validation...")
    if onnx:
        try:
            model_onnx = onnx.load(onnx_path)
            onnx.checker.check_model(model_onnx)
            logger.info("ONNX check passed. The model is valid.")
        except Exception as e:
            logger.error(f"CRITICAL ERROR: ONNX check failed immediately after export: {e}")
            # If the ONNX is invalid, we must stop here.
            raise RuntimeError(f"ONNX validation failed for {onnx_path}")
    else:
        logger.info("Skipping ONNX validation (library not installed).")

# Helper function for building the TensorRT engine from ONNX
def build_tensorrt_engine(onnx_path, engine_path, optimization_batch_size=1024, max_batch_size=1024, fp16_mode=True, workspace_gb=4):
    """Builds a TensorRT engine from an ONNX file with optimization profiles."""
    if trt is None:
        raise ImportError("TensorRT is required for engine building.")
        
    if not os.path.exists(onnx_path):
        logger.error(f"ONNX file not found: {onnx_path}")
        return

    logger.info(f"Building TensorRT engine from {onnx_path}...")
    logger.info(f"Target: {engine_path}. Config: OptBatch={optimization_batch_size}, MaxBatch={max_batch_size}, FP16={fp16_mode}, Workspace={workspace_gb}GB")

    # 1. Initialize builder, network, parser, and config
    builder = trt.Builder(TRT_LOGGER)
    # EXPLICIT_BATCH flag is required for ONNX models
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # Set workspace size.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    # 2. Configure precision
    if fp16_mode:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("Enabled FP16 precision.")
        else:
            logger.warning("WARNING: FP16 mode requested but not fully supported by the platform. Proceeding with FP32.")

    # 3. Parse ONNX model
    logger.info("Starting ONNX parsing...")
    if not parser.parse_from_file(onnx_path):
        logger.error('ERROR: Failed to parse the ONNX file.')
        for error in range(parser.num_errors):
            logger.error(parser.get_error(error))
        return
    logger.info("ONNX parsing finished successfully.")

    # 4. Define optimization profile for dynamic shapes (batch size)
    profile = builder.create_optimization_profile()

    # Calculate MIN/OPT/MAX batch sizes
    min_batch = 1
    # Ensure OPT is between MIN and MAX
    opt_batch = max(min_batch, min(max_batch_size, optimization_batch_size))

    # Iterate through inputs and configure the profile correctly for both dynamic and fixed inputs.
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_shape = list(input_tensor.shape)

        # Check if the first dimension is dynamic (marked as -1 or a string during ONNX export)
        if input_shape[0] == -1 or isinstance(input_shape[0], str):
            # Dynamic Input (e.g., the Actor MLP)
            min_shape = (min_batch,) + tuple(input_shape[1:])
            opt_shape = (opt_batch,) + tuple(input_shape[1:])
            max_shape = (max_batch_size,) + tuple(input_shape[1:])

            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            logger.info(f"Configured dynamic profile for input '{input_tensor.name}': MIN={min_shape}, OPT={opt_shape}, MAX={max_shape}")
        else:
            # Fixed Input (e.g., the U-Net Feature Extractor with B=1)
            # CRITICAL FIX: Explicitly ensure MIN=OPT=MAX=Fixed Shape in the profile.
            fixed_shape = tuple(input_shape)
            profile.set_shape(input_tensor.name, fixed_shape, fixed_shape, fixed_shape)
            logger.info(f"Configured fixed profile for input '{input_tensor.name}': Shape={fixed_shape}")

    # Add the configured profile to the builder config.
    config.add_optimization_profile(profile)

    # 5. Build the engine
    logger.info("Starting TensorRT engine build. This may take several minutes. Monitoring verbose logs...")
    logger.info("If the process crashes silently after this point, it indicates a Segmentation Fault in the native TensorRT library.")
    start_time = time.time()
    
    # Use build_serialized_network
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        # The builder (TRT_LOGGER) will have already logged the specific error (like the API Usage Error seen before)
        logger.error("ERROR: Failed to build the TensorRT engine.")
        return

    logger.info(f"Engine built successfully in {time.time() - start_time:.2f} seconds.")

    # 6. Serialize the engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    logger.info(f"Engine saved to {engine_path}")