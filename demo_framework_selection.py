#!/usr/bin/env python3
"""
Demo script showing how to use the cymetric framework selection features.

This script demonstrates:
1. Default behavior (TensorFlow preferred)
2. Environment variable control
3. Runtime framework switching
"""

def demo_default_behavior():
    """Demonstrate default behavior - TensorFlow is preferred when both are available."""
    print("=== Demo 1: Default Behavior ===")
    import cymetric
    print(f"Default preferred framework: {cymetric.PREFERRED_FRAMEWORK}")
    
    from cymetric.models.measures import ricci_measure
    print(f"ricci_measure imported from: {ricci_measure.__module__}")
    print()

def demo_environment_variable():
    """Demonstrate using environment variable to force PyTorch."""
    print("=== Demo 2: Environment Variable Control ===")
    import os
    # Set environment variable before importing cymetric
    os.environ['CYMETRIC_FRAMEWORK'] = 'torch'
    
    # Import cymetric after setting the environment variable
    import importlib
    import sys
    
    # Clear cymetric modules to force reload
    modules_to_clear = [k for k in sys.modules.keys() if k.startswith('cymetric')]
    for module in modules_to_clear:
        del sys.modules[module]
    
    import cymetric
    print(f"With CYMETRIC_FRAMEWORK=torch: {cymetric.PREFERRED_FRAMEWORK}")
    
    from cymetric.models.measures import ricci_measure
    print(f"ricci_measure imported from: {ricci_measure.__module__}")
    print()

def demo_runtime_switching():
    """Demonstrate runtime framework switching."""
    print("=== Demo 3: Runtime Framework Switching ===")
    import cymetric
    print(f"Initial framework: {cymetric.PREFERRED_FRAMEWORK}")
    
    # Import with default framework
    from cymetric.models.measures import ricci_measure
    print(f"First import from: {ricci_measure.__module__}")
    
    # Switch to the other framework
    other_framework = 'torch' if cymetric.PREFERRED_FRAMEWORK == 'tensorflow' else 'tensorflow'
    cymetric.set_preferred_framework(other_framework)
    print(f"Switched to: {cymetric.get_preferred_framework()}")
    
    # Import again (modules are cleared automatically)
    from cymetric.models.measures import ricci_measure
    print(f"Second import from: {ricci_measure.__module__}")
    print()

if __name__ == "__main__":
    print("Cymetric Framework Selection Demo")
    print("=" * 40)
    
    # Check what frameworks are available
    import cymetric
    print(f"PyTorch available: {cymetric.TORCH_AVAILABLE}")
    print(f"TensorFlow available: {cymetric.TENSORFLOW_AVAILABLE}")
    print()
    
    if cymetric.TORCH_AVAILABLE and cymetric.TENSORFLOW_AVAILABLE:
        print("Both frameworks available - running all demos")
        demo_default_behavior()
        demo_runtime_switching()
        # Note: Environment variable demo is commented out as it requires a fresh Python process
        print("Note: Environment variable demo requires a fresh Python process")
    else:
        print("Only one framework available - demonstrating basic usage")
        demo_default_behavior()
