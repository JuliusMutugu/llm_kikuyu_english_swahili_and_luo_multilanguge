"""
Demo script for Modern LLM concepts (CPU-only version)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Any


class SimpleLLMDemo:
    """Simplified demonstration of modern LLM concepts"""
    
    def __init__(self):
        print("🚀 Modern LLM Concepts Demo")
        print("=" * 40)
        
    def demonstrate_attention_scaling(self):
        """Demonstrate attention mechanism scaling"""
        print("\n🔍 Attention Mechanism Scaling")
        print("-" * 30)
        
        seq_lengths = [128, 256, 512, 1024, 2048, 4096]
        standard_complexity = []
        optimized_complexity = []
        
        for seq_len in seq_lengths:
            # Standard attention: O(n²)
            standard_ops = seq_len ** 2
            standard_complexity.append(standard_ops)
            
            # Flash attention approximation: O(n)
            optimized_ops = seq_len * np.log(seq_len)
            optimized_complexity.append(optimized_ops)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.plot(seq_lengths, standard_complexity, 'r-o', label='Standard Attention O(n²)')
        plt.plot(seq_lengths, optimized_complexity, 'g-o', label='Flash Attention O(n log n)')
        plt.xlabel('Sequence Length')
        plt.ylabel('Computational Operations')
        plt.title('Attention Mechanism Scaling Comparison')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"✅ At sequence length 4096:")
        print(f"   Standard attention: {standard_complexity[-1]:,} operations")
        print(f"   Flash attention: {optimized_complexity[-1]:,.0f} operations")
        print(f"   Speedup: {standard_complexity[-1]/optimized_complexity[-1]:.1f}x")
    
    def demonstrate_quantization_benefits(self):
        """Demonstrate quantization benefits"""
        print("\n🔢 Quantization Benefits")
        print("-" * 25)
        
        # Model sizes for different precisions
        precisions = ['FP32', 'FP16', 'INT8', 'INT4']
        bytes_per_param = [4, 2, 1, 0.5]
        
        # Example: 7B parameter model
        num_params = 7_000_000_000
        model_sizes_gb = [(num_params * bytes_per_param[i]) / (1024**3) 
                         for i in range(len(precisions))]
        
        # Accuracy retention (approximate)
        accuracy_retention = [100, 99.5, 98, 95]
        
        print("Model Size Comparison (7B parameters):")
        print("-" * 40)
        for i, precision in enumerate(precisions):
            print(f"{precision:5}: {model_sizes_gb[i]:5.1f} GB | {accuracy_retention[i]:5.1f}% accuracy")
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Model sizes
        bars1 = ax1.bar(precisions, model_sizes_gb, color=['red', 'orange', 'green', 'blue'])
        ax1.set_title('Model Size by Precision')
        ax1.set_ylabel('Size (GB)')
        for bar, size in zip(bars1, model_sizes_gb):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{size:.1f}GB', ha='center', va='bottom')
        
        # Accuracy retention
        bars2 = ax2.bar(precisions, accuracy_retention, color=['red', 'orange', 'green', 'blue'])
        ax2.set_title('Accuracy Retention')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_ylim(90, 101)
        for bar, acc in zip(bars2, accuracy_retention):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{acc}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_lora_efficiency(self):
        """Demonstrate LoRA parameter efficiency"""
        print("\n🎯 LoRA Parameter Efficiency")
        print("-" * 30)
        
        # Example model parameters
        hidden_size = 4096
        num_layers = 32
        
        # Calculate parameters for different components
        attention_params = num_layers * 4 * (hidden_size * hidden_size)  # Q, K, V, O projections
        ffn_params = num_layers * 2 * (hidden_size * hidden_size * 4)    # FFN layers
        
        total_params = attention_params + ffn_params
        
        # LoRA parameters for different ranks
        ranks = [4, 8, 16, 32, 64]
        lora_params = []
        
        for rank in ranks:
            # LoRA adds rank * (input_dim + output_dim) parameters per linear layer
            # We'll apply to attention projections only
            params_per_layer = 4 * rank * (hidden_size + hidden_size)  # Q, K, V, O
            total_lora = num_layers * params_per_layer
            lora_params.append(total_lora)
        
        print(f"Original model parameters: {total_params:,}")
        print("\nLoRA Parameters by Rank:")
        print("-" * 25)
        
        for rank, params in zip(ranks, lora_params):
            reduction = (params / total_params) * 100
            print(f"Rank {rank:2d}: {params:8,} parameters ({reduction:.3f}% of original)")
        
        # Visualize
        plt.figure(figsize=(10, 6))
        plt.bar(ranks, lora_params, alpha=0.7, color='blue')
        plt.axhline(y=total_params, color='red', linestyle='--', 
                   label=f'Original Model ({total_params:,} params)')
        plt.xlabel('LoRA Rank')
        plt.ylabel('Parameters')
        plt.title('LoRA Parameter Count vs Rank')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Add percentage labels
        for rank, params in zip(ranks, lora_params):
            percentage = (params / total_params) * 100
            plt.text(rank, params * 1.1, f'{percentage:.2f}%', 
                    ha='center', va='bottom')
        
        plt.show()
    
    def demonstrate_moe_efficiency(self):
        """Demonstrate Mixture of Experts efficiency"""
        print("\n🏗️ Mixture of Experts Efficiency")
        print("-" * 35)
        
        # Compare dense vs MoE models
        hidden_size = 4096
        intermediate_size = hidden_size * 4
        num_layers = 32
        
        # Dense model FFN parameters
        dense_ffn_params = num_layers * 2 * (hidden_size * intermediate_size)
        
        # MoE parameters for different expert counts
        expert_counts = [4, 8, 16, 32, 64]
        experts_per_token = 2
        
        moe_params = []
        active_params = []
        
        for num_experts in expert_counts:
            # Total MoE parameters
            total_moe = num_layers * num_experts * 2 * (hidden_size * intermediate_size)
            moe_params.append(total_moe)
            
            # Active parameters (only experts_per_token are used)
            active = num_layers * experts_per_token * 2 * (hidden_size * intermediate_size)
            active_params.append(active)
        
        print(f"Dense model FFN parameters: {dense_ffn_params:,}")
        print(f"Experts per token: {experts_per_token}")
        print("\nMoE Model Comparison:")
        print("-" * 30)
        
        for i, num_experts in enumerate(expert_counts):
            total = moe_params[i]
            active = active_params[i]
            efficiency = active / total * 100
            
            print(f"{num_experts:2d} experts: {total:10,} total | {active:10,} active | {efficiency:.1f}% efficiency")
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Total vs Active parameters
        x = np.arange(len(expert_counts))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, moe_params, width, label='Total Parameters', alpha=0.7)
        bars2 = ax1.bar(x + width/2, active_params, width, label='Active Parameters', alpha=0.7)
        ax1.axhline(y=dense_ffn_params, color='red', linestyle='--', 
                   label='Dense Model')
        
        ax1.set_xlabel('Number of Experts')
        ax1.set_ylabel('Parameters')
        ax1.set_title('MoE vs Dense Model Parameters')
        ax1.set_xticks(x)
        ax1.set_xticklabels(expert_counts)
        ax1.legend()
        ax1.set_yscale('log')
        
        # Efficiency
        efficiency_ratios = [active / total * 100 for active, total in zip(active_params, moe_params)]
        ax2.bar(expert_counts, efficiency_ratios, color='green', alpha=0.7)
        ax2.set_xlabel('Number of Experts')
        ax2.set_ylabel('Parameter Efficiency (%)')
        ax2.set_title('MoE Parameter Efficiency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def benchmark_comparison(self):
        """Show modern LLM benchmark comparison"""
        print("\n📊 Modern LLM Benchmark Comparison")
        print("-" * 40)
        
        # Simulated benchmark data for different model types
        models = ['GPT-3.5', 'GPT-4', 'LLaMA-7B', 'LLaMA-13B', 'Claude-3', 'Gemini-Pro']
        
        benchmarks = {
            'MMLU': [70.0, 86.4, 35.1, 46.9, 86.8, 80.0],
            'GSM8K': [57.1, 92.0, 11.0, 17.8, 95.0, 86.5],
            'HumanEval': [48.1, 67.0, 10.5, 15.8, 71.2, 67.7],
            'HellaSwag': [85.5, 95.3, 76.1, 79.2, 95.4, 87.8]
        }
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, (benchmark, scores) in enumerate(benchmarks.items()):
            ax = axes[i]
            bars = ax.bar(models, scores, color=colors)
            ax.set_title(f'{benchmark} Benchmark Scores')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add score labels
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{score:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary table
        print("\nBenchmark Summary Table:")
        print("-" * 60)
        print(f"{'Model':<12} | {'MMLU':<6} | {'GSM8K':<6} | {'HumanEval':<9} | {'HellaSwag'}")
        print("-" * 60)
        
        for i, model in enumerate(models):
            print(f"{model:<12} | {benchmarks['MMLU'][i]:5.1f} | "
                  f"{benchmarks['GSM8K'][i]:5.1f} | "
                  f"{benchmarks['HumanEval'][i]:8.1f} | "
                  f"{benchmarks['HellaSwag'][i]:5.1f}")
    
    def run_all_demos(self):
        """Run all demonstrations"""
        print("🎬 Running all Modern LLM demonstrations...")
        print("=" * 50)
        
        demos = [
            ("Attention Scaling", self.demonstrate_attention_scaling),
            ("Quantization Benefits", self.demonstrate_quantization_benefits),
            ("LoRA Efficiency", self.demonstrate_lora_efficiency),
            ("MoE Efficiency", self.demonstrate_moe_efficiency),
            ("Benchmark Comparison", self.benchmark_comparison)
        ]
        
        for name, demo_func in demos:
            try:
                demo_func()
                print(f"✅ {name} completed")
            except Exception as e:
                print(f"⚠️ {name} failed: {e}")
            
            print()
        
        print("🎉 All demonstrations completed!")
        print("\n💡 Key Takeaways:")
        print("   ✅ Modern optimizations provide significant improvements")
        print("   ✅ Different techniques solve different problems")
        print("   ✅ Combining optimizations maximizes benefits")
        print("   ✅ Choose techniques based on your specific constraints")


def main():
    """Main demo function"""
    demo = SimpleLLMDemo()
    
    print("\nSelect a demonstration:")
    print("1. Attention Mechanism Scaling")
    print("2. Quantization Benefits")
    print("3. LoRA Parameter Efficiency")
    print("4. Mixture of Experts Efficiency")
    print("5. LLM Benchmark Comparison")
    print("6. Run All Demonstrations")
    
    try:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            demo.demonstrate_attention_scaling()
        elif choice == '2':
            demo.demonstrate_quantization_benefits()
        elif choice == '3':
            demo.demonstrate_lora_efficiency()
        elif choice == '4':
            demo.demonstrate_moe_efficiency()
        elif choice == '5':
            demo.benchmark_comparison()
        elif choice == '6':
            demo.run_all_demos()
        else:
            print("Invalid choice. Running all demonstrations...")
            demo.run_all_demos()
            
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        print("Running all demonstrations as fallback...")
        demo.run_all_demos()


if __name__ == "__main__":
    main()
