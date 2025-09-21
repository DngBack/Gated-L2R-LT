"""Quick diagnostic script to understand the current state."""
import torch
import json
import os

def check_checkpoint_structure(checkpoint_path):
    """Check what's inside a checkpoint."""
    print(f"\\n=== Checking {checkpoint_path} ===")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ File not found: {checkpoint_path}")
        return
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("✅ Checkpoint loaded successfully")
        
        print("Keys in checkpoint:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], torch.Tensor):
                print(f"  {key}: {checkpoint[key].shape} ({checkpoint[key].dtype})")
            else:
                print(f"  {key}: {type(checkpoint[key])}")
                
        # Check specific values
        if 'alpha' in checkpoint:
            alpha = checkpoint['alpha']
            print(f"\\nAlpha values: {alpha}")
            print(f"Alpha sum: {alpha.sum():.4f}")
            
        if 'mu' in checkpoint:
            mu = checkpoint['mu']
            print(f"Mu values: {mu}")
            
        if 'gating' in checkpoint:
            print(f"\\nGating state_dict keys: {list(checkpoint['gating'].keys())}")
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")


def check_expert_checkpoints(base_dir):
    """Check expert checkpoints."""
    experts = ['head', 'tail', 'balanced']
    
    print("\\n=== Checking Expert Checkpoints ===")
    for expert in experts:
        path = os.path.join(base_dir, f'expert_{expert}', 'best.pt')
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location='cpu')
                print(f"✅ {expert}: {list(checkpoint.keys())}")
            except Exception as e:
                print(f"❌ {expert}: Error - {e}")
        else:
            print(f"❌ {expert}: File not found")


def analyze_log_files(base_dir):
    """Analyze log files to understand training progress."""
    print("\\n=== Analyzing Log Files ===")
    
    for folder in ['expert_head', 'expert_tail', 'expert_balanced', 'gating', 'worst_group']:
        log_path = os.path.join(base_dir, folder, 'log.jsonl')
        
        if not os.path.exists(log_path):
            print(f"❌ {folder}: No log file")
            continue
            
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                print(f"⚠️  {folder}: Empty log file")
                continue
                
            # Parse first and last entries
            first_entry = json.loads(lines[0])
            last_entry = json.loads(lines[-1])
            
            print(f"\\n📁 {folder}:")
            print(f"  Total entries: {len(lines)}")
            
            if 'loss' in first_entry and 'loss' in last_entry:
                print(f"  Loss: {first_entry['loss']:.4f} → {last_entry['loss']:.4f}")
                
            if 'coverage' in last_entry:
                print(f"  Final coverage: {last_entry['coverage']:.4f}")
                
            if 'selective_risk' in last_entry:
                print(f"  Final selective risk: {last_entry['selective_risk']:.4f}")
                
            if 'balanced_error' in last_entry:
                print(f"  Final balanced error: {last_entry['balanced_error']:.4f}")
                
            if 'worst_group_error' in last_entry:
                print(f"  Final worst group error: {last_entry['worst_group_error']:.4f}")
                
        except Exception as e:
            print(f"❌ {folder}: Error reading log - {e}")


def main():
    print("🔍 DIAGNOSTIC ANALYSIS")
    print("=" * 50)
    
    base_dir = "src/outputs/cifar100lt"
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"❌ Base directory not found: {base_dir}")
        return
    
    # Check expert checkpoints
    check_expert_checkpoints(base_dir)
    
    # Check gating checkpoints
    gating_checkpoint = os.path.join(base_dir, "gating", "gating.pt")
    check_checkpoint_structure(gating_checkpoint)
    
    worst_group_checkpoint = os.path.join(base_dir, "worst_group", "worst_group.pt")
    check_checkpoint_structure(worst_group_checkpoint)
    
    # Analyze logs
    analyze_log_files(base_dir)
    
    print("\\n🏁 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    print("Based on the logs, your current issues are:")
    print("1. High worst-group error (~67%)")
    print("2. High balanced error (~47-49%)")
    print("3. Reasonable coverage (~98%)")
    print("\\nThis suggests:")
    print("- Experts might not be well-calibrated")
    print("- Gating network is not learning proper routing")
    print("- Plugin parameters (α, μ) might be suboptimal")
    print("\\nRecommended fixes:")
    print("1. Check expert individual performance")
    print("2. Try improved gating implementation")
    print("3. Adjust abstention cost parameter")


if __name__ == "__main__":
    main()