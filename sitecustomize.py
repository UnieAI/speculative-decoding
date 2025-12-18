import importlib.util
import os
import sys
from typing import Optional


def register_custom_algorithm_silent():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        register_module_path = os.path.join(current_dir, "register_custom_algorithm.py")

        if os.path.exists(register_module_path):
            spec = importlib.util.spec_from_file_location(
                "register_custom_algorithm", register_module_path
            )
            if spec and spec.loader:
                register_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(register_module)

                if hasattr(register_module, "register_custom_ngram_algorithm"):
                    register_module.register_custom_ngram_algorithm(
                        force_reregister=True,
                    )

    except Exception:
        pass


def check_and_register_custom_algorithms():
    try:
        if any(key.startswith("SGLANG_") for key in os.environ.keys()):
            register_custom_algorithm_silent()
        elif "sglang" in sys.modules:
            register_custom_algorithm_silent()
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if os.path.exists(
                os.path.join(current_dir, "register_custom_algorithm.py")
            ):
                register_custom_algorithm_silent()

    except Exception:
        pass


if __name__ != "__main__":
    check_and_register_custom_algorithms()


def manual_register():
    register_custom_algorithm_silent()


if __name__ == "__main__":
    print("CustomNGRAM Site Customization - Manual Test")
    print("=" * 45)

    try:
        manual_register()
        print("CustomNGRAM algorithm registration attempted")

        try:
            from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

            algorithm = SpeculativeAlgorithm.from_string("CUSTOM_NGRAM")
            print("CustomNGRAM algorithm successfully registered and accessible")
            print(f"  Algorithm: {algorithm}")
            print(f"  Value: {algorithm.value}")
            print(f"  Is NGRAM: {algorithm.is_ngram()}")
        except ValueError as e:
            print(f"✗ Algorithm registration might have failed: {e}")

    except Exception as e:
        print(f"✗ Registration failed: {e}")
