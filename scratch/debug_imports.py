try:
    import numpy
    print("numpy ok")
    import xgboost
    print("xgboost ok")
    import onnxmltools
    print("onnxmltools ok")
    from skl2onnx.common.data_types import FloatTensorType
    print("skl2onnx ok")
    import onnxruntime
    print("onnxruntime ok")
except Exception as e:
    print(f"FAILED: {e}")
