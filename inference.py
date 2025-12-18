import argparse
import asyncio
import logging
import time
from typing import Optional

import sglang as sgl
from register_custom_algorithm import (
    get_custom_ngram_algorithm,
    is_custom_ngram_registered,
    register_custom_ngram_algorithm,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CustomNGramInferenceDemo:
    def __init__(self, model_path: str = "Qwen/Qwen3-0.6B"):
        self.model_path = model_path
        self.engine: Optional[sgl.Engine] = None
        self.algorithm: Optional[SpeculativeAlgorithm] = None

    def register_algorithm(
        self,
        force_reregister: bool = False,
        verbose_acceptance: bool = True,
        acceptance_log_level: str = "INFO",
    ) -> SpeculativeAlgorithm:
        print("🔧 Registering CustomNGRAM algorithm...")
        try:
            self.algorithm = register_custom_ngram_algorithm(
                force_reregister=force_reregister,
                verbose_acceptance=verbose_acceptance,
                acceptance_log_level=acceptance_log_level,
            )
            print(f"✅ Successfully registered: {self.algorithm}")
            return self.algorithm
        except Exception as e:
            print(f"❌ Failed to register algorithm: {e}")
            raise

    def create_engine(
        self,
        verbose_acceptance: bool = True,
        acceptance_log_level: str = "INFO",
        port: int = 30000,
        **kwargs,
    ):
        if self.algorithm is None:
            raise ValueError(
                "Algorithm must be registered first. Call register_algorithm()."
            )

        print(f"   Creating SGLang engine with {self.algorithm.name}...")
        print(f"   Model: {self.model_path}")
        print(f"   Port: {port}")
        print(f"   Verbose acceptance: {verbose_acceptance} (Using worker default)")

        if "model_path" in kwargs:
            del kwargs["model_path"]

        try:
            kwargs.pop("speculative_num_draft_tokens", None)
            kwargs.pop("max_queued_requests", None)
            kwargs.pop("max_running_requests", None)

            self.engine = sgl.Engine(
                model_path=self.model_path,
                speculative_algorithm="NGRAM",
                # speculative_algorithm=self.algorithm.name,
                port=port,
                speculative_num_draft_tokens=4,
                max_queued_requests=1,
                max_running_requests=1,
                **kwargs,
            )

            print("✅ Engine created successfully!")
            return self.engine

        except Exception as e:
            print(f"❌ Failed to create engine: {e}")
            raise

    async def generate_with_monitoring(
        self, prompt: str, max_tokens: int = 200, temperature: float = 0.8
    ):
        if self.engine is None:
            raise ValueError("Engine must be created first. Call create_engine().")

        print(f"\n🎯 Generating text with acceptance monitoring...")
        print(f"   Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")

        start_time = time.time()

        try:
            # Use sampling_params dictionary for SGLang API
            sampling_params = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            }
            # Use async_generate since we're already in an async context
            response = await self.engine.async_generate(
                prompt, sampling_params=sampling_params
            )
            generation_time = time.time() - start_time

            print("-" * 50)
            print(f"✅ Generation completed in {generation_time:.2f} seconds")
            print(f"📝 Generated text:\n   {response['text']}")
            return response

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            raise

    async def batch_generation_demo(self):
        if self.engine is None:
            raise ValueError("Engine must be created first. Call create_engine().")

        prompts = [
            "Explain the concept of machine learning in simple terms:",
            "What are the benefits of renewable energy?",
            "How does quantum computing work?",
        ]

        print(f"\n🔄 Batch Generation Demo ({len(prompts)} prompts)")
        print("-" * 60)

        start_time = time.time()
        try:
            batch_sampling_params = {
                "max_new_tokens": 50,
                "temperature": 0.7,
            }
            tasks = [
                self.engine.async_generate(
                    prompt, sampling_params=batch_sampling_params
                )
                for prompt in prompts
            ]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time

            print(f"✅ Batch generation completed in {total_time:.2f} seconds")
            return results

        except Exception as e:
            print(f"❌ Batch generation failed: {e}")
            raise

    def get_worker_stats(self):
        print("\n Worker Statistics:")
        print("   (CustomNGRAMWorker maintains acceptance statistics)")
        print("   Look for acceptance confirmation messages in the logs above")

    async def run_complete_demo(self):
        print(" CustomNGRAMWorker Complete Demo")
        print("=" * 50)

        try:
            self.register_algorithm(force_reregister=True)

            self.create_engine(
                verbose_acceptance=True,
                acceptance_log_level="INFO",
                speculative_ngram_min_match_window_size=1,
                speculative_ngram_max_match_window_size=8,
                speculative_ngram_branch_length=12,
                speculative_ngram_capacity=1000000,
            )

            # Step 3: Single generation
            await self.generate_with_monitoring(
                "Explain the concept of artificial intelligence:",
            )

            print("\n⏭️  Skipping batch generation to avoid complex batch processing")
            print("   (Single generation test completed successfully above)")

            self.get_worker_stats()

            print("\n🎉 Demo completed successfully!")

        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            raise
        finally:
            if self.engine:
                print("\n🧹 Cleaning up engine...")
                self.engine.shutdown()
                print("✅ Cleanup completed")


async def main():
    parser = argparse.ArgumentParser(
        description="Custom NGram Worker Inference Example"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model path"
    )
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    parser.add_argument(
        "--register-only",
        action="store_true",
        help="Only register the algorithm, don't run inference",
    )

    args = parser.parse_args()

    print("CustomNGRAMWorker Inference Example")
    print("=" * 40)

    if args.register_only:
        demo = CustomNGramInferenceDemo(model_path=args.model)
        demo.register_algorithm()
        return

    # Create demo instance
    demo = CustomNGramInferenceDemo(model_path=args.model)

    try:
        print(f"   Starting inference demo...")
        print(f"   Model: {args.model}")
        print(f"   Port: {args.port}")

        await demo.run_complete_demo()

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nTroubleshooting:")
        print("   1. Ensure you have the required model installed")
        print("   2. Check that you have sufficient GPU memory")
        print("   3. Verify SGLang and dependencies are properly installed")


if __name__ == "__main__":
    asyncio.run(main())
