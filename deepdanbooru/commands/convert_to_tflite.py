from typing import List
import torch
import deepdanbooru as dd

def convert_to_torchscript_from_model(
    project_path: str, model_path: str, save_path: str,
    optimize: bool = True,
    verbose: bool = False
):
    """Convert PyTorch model to TorchScript for deployment"""
    if not model_path and not project_path:
        raise Exception("You must provide project path or model path.")

    if not save_path:
        raise Exception("You must provide a path to save TorchScript model.")

    if model_path:
        if verbose:
            print(f"Loading model from {model_path} ...")
        model = torch.load(model_path, map_location='cpu')
    else:
        if verbose:
            print(f"Loading model from project {project_path} ...")
        model = dd.project.load_model_from_project(project_path)

    model.eval()

    if verbose:
        print("Converting to TorchScript ...")

    # Create example input for tracing
    example_input = torch.randn(1, 3, 299, 299)  # Adjust dimensions as needed
    
    # Convert to TorchScript using tracing
    traced_model = torch.jit.trace(model, example_input)
    
    if optimize:
        traced_model = torch.jit.optimize_for_inference(traced_model)

    if verbose:
        print("Saving ...")

    traced_model.save(save_path)

    if verbose:
        print(f"TorchScript model has been saved to {save_path}")

# Keep the old function name for compatibility
def convert_to_tflite_from_from_saved_model(
    project_path: str, model_path: str, save_path: str,
    optimizations: List = None,
    verbose: bool = False
):
    """Legacy function name - now converts to TorchScript instead of TFLite"""
    print("Warning: TFLite conversion not available. Converting to TorchScript instead.")
    
    # Adjust save path extension
    if save_path.endswith('.tflite'):
        save_path = save_path.replace('.tflite', '.pt')
    
    convert_to_torchscript_from_model(
        project_path=project_path,
        model_path=model_path, 
        save_path=save_path,
        optimize=True,
        verbose=verbose
    )
