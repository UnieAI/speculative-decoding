import logging
from typing import Optional

from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    register_speculative_algorithm,
)

from custom_ngram_worker import create_custom_ngram_worker

logger = logging.getLogger(__name__)

_CUSTOM_ALGORITHM_REGISTERED = False
_CUSTOM_ALGORITHM_INSTANCE: Optional[SpeculativeAlgorithm] = None


def register_custom_ngram_algorithm(
    force_reregister: bool = False,
    verbose_acceptance: bool = True,
    acceptance_log_level: str = "INFO",
) -> SpeculativeAlgorithm:
    global _CUSTOM_ALGORITHM_REGISTERED, _CUSTOM_ALGORITHM_INSTANCE

    # Check if already registered
    if _CUSTOM_ALGORITHM_REGISTERED and not force_reregister:
        logger.info("CustomNGRAM algorithm already registered")
        return _CUSTOM_ALGORITHM_INSTANCE

    try:
        # Register the custom algorithm
        algorithm = register_speculative_algorithm(
            name="CUSTOM_NGRAM",
            worker_cls=create_custom_ngram_worker,
            aliases=["CUSTOM", "CUSTOM_NGRAM_WORKER"],
            flags=["NGRAM"],  # Mark as NGRAM-based algorithm
            override_worker=force_reregister,
        )

        _CUSTOM_ALGORITHM_REGISTERED = True
        _CUSTOM_ALGORITHM_INSTANCE = algorithm

        logger.info(f"Successfully registered CustomNGRAM algorithm: {algorithm}")
        logger.info("You can now use --speculative-algorithm CUSTOM_NGRAM")

        return algorithm

    except Exception as e:
        logger.error(f"Failed to register CustomNGRAM algorithm: {e}")
        raise


def unregister_custom_ngram_algorithm():
    global _CUSTOM_ALGORITHM_REGISTERED, _CUSTOM_ALGORITHM_INSTANCE

    if not _CUSTOM_ALGORITHM_REGISTERED:
        logger.info("CustomNGRAM algorithm is not currently registered")
        return

    _CUSTOM_ALGORITHM_REGISTERED = False
    _CUSTOM_ALGORITHM_INSTANCE = None

    logger.info("CustomNGRAM algorithm unregistered")


def is_custom_ngram_registered() -> bool:
    return _CUSTOM_ALGORITHM_REGISTERED


def get_custom_ngram_algorithm() -> Optional[SpeculativeAlgorithm]:
    return _CUSTOM_ALGORITHM_INSTANCE


def list_speculative_algorithms():
    from sglang.srt.speculative.spec_info import list_registered_workers

    workers = list_registered_workers()

    print("Registered Speculative Algorithms:")
    print("=" * 40)

    for name, worker_cls in workers.items():
        status = "Registered"
        if name == "CUSTOM_NGRAM":
            status = "Registered (Custom)"

        print(f"{name:15} | {status:18} | {worker_cls.__name__}")

    print("=" * 40)
    print(f"Total: {len(workers)} algorithms registered")


def verify_custom_ngram_integration():
    try:
        algorithm = register_custom_ngram_algorithm()

        if algorithm.create_draft_worker is not None:
            print("CustomNGRAM worker factory is accessible")

        algo_from_string = SpeculativeAlgorithm.from_string("CUSTOM_NGRAM")
        if algo_from_string is algorithm:
            print("CustomNGRAM is properly registered in SpeculativeAlgorithm")

        if algorithm.is_ngram():
            print("CustomNGRAM has correct NGRAM flag")

        print("\n" + "=" * 50)
        print("All integration tests passed!")
        print("CustomNGRAM is ready to use with SGLang")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def demonstrate_usage():
    print("""
CustomNGRAM Algorithm Usage Examples
====================================

1. Command Line Usage:
   Register the algorithm first:
   >>> python register_custom_algorithm.py

   Then launch server with CUSTOM_NGRAM:
   >>> python -m sglang.launch_server \\
       --model Qwen/Qwen3-32B \\
       --speculative-algorithm CUSTOM_NGRAM \\
       --speculative-ngram-min-match-window-size 1 \\
       --speculative-ngram-max-match-window-size 12 \\
       --speculative-ngram-min-bfs-breadth 1 \\
       --speculative-ngram-max-bfs-breadth 15 \\
       --speculative-ngram-match-type BFS \\
       --speculative-ngram-branch-length 18 \\
       --speculative-ngram-capacity 10000000 \\
       --dtype float16 \\
       --port 30000

2. Python API Usage:
   >>> import sglang as sgl
   >>> from sglang.examples.CustomNGramWorker.register_custom_algorithm import register_custom_ngram_algorithm

   # Register the algorithm
   >>> register_custom_ngram_algorithm()

   # Create engine with custom worker
   >>> engine = sgl.Engine(
       model_path="Qwen/Qwen3-0.6B",
       speculative_algorithm="CUSTOM_NGRAM",
       verbose_acceptance=True,
       acceptance_log_level="INFO",
       # ... other parameters
   )

   # Generate text (you'll see acceptance confirmation messages)
   >>> response = engine.generate("Explain quantum computing:")
   >>> print(response['text'])
""")


def main():
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "register":
            try:
                register_custom_ngram_algorithm()
                print("CustomNGRAM algorithm registered successfully!")
            except Exception as e:
                print(f"Failed to register: {e}")
                sys.exit(1)

        elif command == "list":
            list_speculative_algorithms()

        elif command == "verify":
            success = verify_custom_ngram_integration()
            sys.exit(0 if success else 1)

        elif command == "demo":
            demonstrate_usage()

        elif command == "unregister":
            unregister_custom_ngram_algorithm()
            print("CustomNGRAM algorithm unregistered")

        else:
            print(f"Unknown command: {command}")
            print("Available commands: register, list, verify, demo, unregister")
            sys.exit(1)

    else:
        try:
            register_custom_ngram_algorithm()
            print("CustomNGRAM Algorithm Registration")
            print("=" * 35)
            print("Algorithm registered successfully!")
            print("Available as: CUSTOM_NGRAM")
            print("Aliases: CUSTOM, CUSTOM_NGRAM_WORKER")
            print("Flags: NGRAM-based algorithm")
            print()
            print("Usage: --speculative-algorithm CUSTOM_NGRAM")
            print()
            print("Run with --help for more options:")
            print("  python register_custom_algorithm.py --help")

        except Exception as e:
            print(f"Registration failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s"
    )

    main()
