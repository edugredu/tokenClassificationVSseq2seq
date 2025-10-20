#!/usr/bin/env python3
import sys
import json
from transformers import AutoTokenizer


def analyze_model_lengths(train_path, valid_path, model_path, template_path):
    """
    Analyze both input and output lengths for optimal model configuration.
    Returns recommendations for max_seq_len and eval_max_new_tokens.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    
    # Parse template (reuse your existing function)
    def parse_template(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        import re
        sys_match = re.search(r"###\s*SYSTEM\s*###\s*(.*?)(?=###\s*USER\s*###|$)", content, flags=re.S | re.I)
        usr_match = re.search(r"###\s*USER\s*###\s*(.*)$", content, flags=re.S | re.I)
        
        system_tmpl = sys_match.group(1).strip() if sys_match else ""
        user_tmpl = usr_match.group(1).strip() if usr_match else ""
        return {"system": system_tmpl, "user": user_tmpl}
    
    tmpl = parse_template(template_path)
    
    # Storage for length analysis
    input_lengths = []      # Full input context (system + user)
    output_lengths = []     # Assistant responses only  
    total_lengths = []      # Input + Output combined
    
    for file_path in [train_path, valid_path]:
        print(f"Analyzing {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line)
                    
                    # Extract components
                    user_input = example.get("user", "")
                    raw_asst = example.get("assistant", "")
                    
                    # Handle assistant response
                    if isinstance(raw_asst, (list, dict)):
                        asst_text = json.dumps(raw_asst, ensure_ascii=False)
                    else:
                        asst_text = str(raw_asst).strip()
                    
                    # Build full input context using template
                    system_text = tmpl["system"] if tmpl["system"] else ""
                    user_text = tmpl["user"].replace("{{USER_MESSAGE}}", user_input) if "{{USER_MESSAGE}}" in tmpl["user"] else user_input
                    
                    # Create full input context (what model sees as input)
                    full_input = ""
                    if system_text:
                        full_input += system_text + "\n"
                    full_input += user_text
                    
                    # Tokenize components
                    input_tokens = tokenizer.encode(full_input, add_special_tokens=True)
                    output_tokens = tokenizer.encode(asst_text, add_special_tokens=False)
                    
                    input_len = len(input_tokens)
                    output_len = len(output_tokens)
                    total_len = input_len + output_len
                    
                    input_lengths.append(input_len)
                    output_lengths.append(output_len)
                    total_lengths.append(total_len)
                    
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
    
    # Analyze and report results
    if input_lengths and output_lengths:
        print(f"\n=== MODEL LENGTH ANALYSIS ===")
        print(f"Total samples analyzed: {len(input_lengths)}")
        
        # Input Length Analysis
        print(f"\n--- INPUT LENGTH ANALYSIS (Context) ---")
        print_stats("Input", input_lengths)
        
        # Output Length Analysis  
        print(f"\n--- OUTPUT LENGTH ANALYSIS (Responses) ---")
        print_stats("Output", output_lengths)
        
        # Total Length Analysis
        print(f"\n--- TOTAL LENGTH ANALYSIS (Input + Output) ---")
        print_stats("Total", total_lengths)
        
        # Configuration Recommendations
        print(f"\n=== CONFIGURATION RECOMMENDATIONS ===")
        
        # For max_seq_len (total sequence length)
        sorted_total = sorted(total_lengths)
        total_95th = sorted_total[int(len(sorted_total) * 0.95)]
        total_99th = sorted_total[int(len(sorted_total) * 0.99)]
        total_max = max(total_lengths)
        
        print(f"\nmax_seq_len (total sequence length):")
        print(f"  Conservative (95th percentile): max_seq_len={total_95th}")
        print(f"  Safe (99th percentile): max_seq_len={total_99th}")
        print(f"  Maximum observed: max_seq_len={total_max}")
        print(f"  Recommended: max_seq_len={min(total_99th + 128, 4096)}")  # Add buffer, cap at common limit
        
        # For eval_max_new_tokens (generation length)
        sorted_output = sorted(output_lengths)
        output_95th = sorted_output[int(len(sorted_output) * 0.95)]
        output_99th = sorted_output[int(len(sorted_output) * 0.99)]
        output_max = max(output_lengths)
        
        print(f"\neval_max_new_tokens (generation length):")
        print(f"  Conservative (95th percentile): eval_max_new_tokens={output_95th}")
        print(f"  Safe (99th percentile): eval_max_new_tokens={output_99th}")
        print(f"  Maximum observed: eval_max_new_tokens={output_max}")
        print(f"  Recommended: eval_max_new_tokens={output_99th}")
        
        # Memory and efficiency insights
        print(f"\n--- EFFICIENCY INSIGHTS ---")
        
        # Input distribution
        long_inputs = sum(1 for length in input_lengths if length > 512)
        very_long_inputs = sum(1 for length in input_lengths if length > 1024)
        
        print(f"Inputs over 512 tokens: {long_inputs} ({long_inputs/len(input_lengths)*100:.1f}%)")
        print(f"Inputs over 1024 tokens: {very_long_inputs} ({very_long_inputs/len(input_lengths)*100:.1f}%)")
        
        # Output distribution
        long_outputs = sum(1 for length in output_lengths if length > 128)
        very_long_outputs = sum(1 for length in output_lengths if length > 512)
        
        print(f"Outputs over 128 tokens: {long_outputs} ({long_outputs/len(output_lengths)*100:.1f}%)")
        print(f"Outputs over 512 tokens: {very_long_outputs} ({very_long_outputs/len(output_lengths)*100:.1f}%)")
        
        # Memory estimation (rough)
        avg_total = sum(total_lengths) / len(total_lengths)
        print(f"\nAverage total sequence length: {avg_total:.1f} tokens")
        print(f"Estimated memory per sample (FP16): ~{avg_total * 2 / 1024:.1f} KB")
        
    return {
        'input_lengths': input_lengths,
        'output_lengths': output_lengths, 
        'total_lengths': total_lengths
    }


def print_stats(name, lengths):
    """Helper function to print statistical analysis"""
    print(f"{name} - Min: {min(lengths)}, Max: {max(lengths)}")
    print(f"{name} - Mean: {sum(lengths)/len(lengths):.1f}, Median: {sorted(lengths)[len(lengths)//2]}")
    
    sorted_lengths = sorted(lengths)
    percentiles = [75, 90, 95, 99]
    percentile_str = ", ".join([f"{p}th: {sorted_lengths[int(len(sorted_lengths) * p / 100)]}" 
                               for p in percentiles])
    print(f"{name} - Percentiles - {percentile_str}")


# Run the analysis
if __name__ == "__main__":

    #Get the first argument, that is the model name to be used
    model_name = sys.argv[1] if len(sys.argv) > 1 else "llama-3.2-3b-instruct"

    print(f"Using model: {model_name}")

    results = analyze_model_lengths(
        train_path="Instruction/instructed_prompts_train_p0.jsonl",
        valid_path="Instruction/instructed_prompts_valid_p0.jsonl", 
        model_path=model_name,
        template_path="Instruction/basePromptInstruction.txt"
    )
    
    # Optional: Save results for further analysis
    print(f"\nAnalysis complete. Results available in 'results' dictionary.")