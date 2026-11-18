"""Hailo-8 compilation. Requires Hailo Dataflow Compiler 3.33 on an x86 Linux PC."""
from pathlib import Path

from hailo.calibration import calibration_images


def compile_model(weights, calibration, output):
    from hailo_sdk_client import ClientRunner
    from ultralytics import YOLO

    weights, output = Path(weights), Path(output)
    model = YOLO(str(weights))
    if model.task != "detect" or "26" not in str(model.model.yaml.get("yaml_file", weights.name)):
        raise ValueError("This compiler script uses the YOLO26 detection head.")
    # Export the same six head outputs used by the working HEF on the Pi.
    onnx = model.export(format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False, device="cpu")
    end_nodes = [f"/model.23/cv{branch}.{scale}/cv{branch}.{scale}.2/Conv" for branch in (2, 3) for scale in range(3)]
    runner = ClientRunner(hw_arch="hailo8")
    runner.translate_onnx_model(str(onnx), "grocery_yolo", end_node_names=end_nodes)
    output.mkdir(parents=True, exist_ok=True)
    runner.save_har(str(output / "parsed.har"))
    # Calibrate the weights; optional compiler fine-tuning is disabled.
    runner.load_model_script('normalization1 = normalization([0, 0, 0], [255, 255, 255])\nmodel_optimization_flavor(optimization_level=0)\npost_quantization_optimization(finetune, policy=disabled)\nquantization_param([output_layer1, output_layer2, output_layer3, output_layer4, output_layer5, output_layer6], precision_mode=a16_w16)')
    runner.optimize(calibration_images(calibration))
    runner.save_har(str(output / "quantized.har"))
    (output / "yolo.hef").write_bytes(runner.compile())
    (output / "classes.txt").write_text("\n".join(model.names[index] for index in sorted(model.names)) + "\n")
    return output / "yolo.hef"
