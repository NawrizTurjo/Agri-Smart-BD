
import torch
import numpy as np

def inspect():
    try:
        print("Loading weights...")
        state_dict = torch.load('plant_disease_weights.pth', map_location='cpu')
        
        print(f"Total keys: {len(state_dict)}")
        
        all_means = []
        all_stds = []
        has_zeros = False
        has_nans = False
        
        for k, v in state_dict.items():
            if v.dtype == torch.float32 or v.dtype == torch.float64:
                m = v.mean().item()
                s = v.std().item()
                all_means.append(m)
                all_stds.append(s)
                
                if torch.sum(v == 0).item() == v.numel():
                    print(f"WARNING: Layer {k} is all ZEROS!")
                    has_zeros = True
                
                if torch.isnan(v).any():
                    print(f"WARNING: Layer {k} has NaNs!")
                    has_nans = True
                    
        print(f"Average Weight Mean: {np.mean(all_means):.4f}")
        print(f"Average Weight Std: {np.mean(all_stds):.4f}")
        
        if not has_zeros and not has_nans:
            print("Weights look healthy (non-zero, non-NaN).")
        else:
            print("Weights might be corrupted.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
